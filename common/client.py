"""Async grounding client that load-balances requests across the vLLM server
pool (serving/serve_pool.sh). Sends an image + instruction, returns a point in
[0,1]x[0,1] relative to the image sent.
"""
from __future__ import annotations

import io
import os
import time
import base64
import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import aiohttp
from PIL import Image

from grounding import smart_resize
from models import ModelSpec, get_spec

Image.MAX_IMAGE_PIXELS = None

DEFAULT_PORTS = list(range(8001, 8008))
MODEL_NAME = os.environ.get("GROUNDER_NAME", "grounder")
# resize + JPEG encode are CPU-bound; run them off the event loop so many
# images encode in parallel and the loop stays free to keep the GPUs fed.
_EXEC = ThreadPoolExecutor(max_workers=int(os.environ.get("ENC_WORKERS", "24")))


def endpoints(ports: Optional[List[int]] = None) -> List[str]:
    ports = ports or DEFAULT_PORTS
    return [f"http://127.0.0.1:{p}/v1/chat/completions" for p in ports]


def health_urls(ports: Optional[List[int]] = None) -> List[str]:
    ports = ports or DEFAULT_PORTS
    return [f"http://127.0.0.1:{p}/health" for p in ports]


def _healthy_endpoints(eps: List[str], timeout: float = 4.0) -> List[str]:
    """Synchronously probe /health for each chat endpoint; return the live ones.
    Used at pool construction so a server that failed to launch (or crashed) is
    excluded from routing instead of silently swallowing ~1/N of the requests."""
    import urllib.request
    out = []
    for ep in eps:
        hu = ep.replace("/v1/chat/completions", "/health")
        try:
            with urllib.request.urlopen(hu, timeout=timeout) as r:
                if getattr(r, "status", 200) == 200:
                    out.append(ep)
        except Exception:
            pass
    return out


def prep_b64(spec: ModelSpec, img) -> Tuple[str, int, int]:
    """Resize to the model's smart-resize fixed point and JPEG-encode.
    `img` may be a PIL image or a file path (opened here, off the event loop).
    Returns (base64, resized_w, resized_h). CPU-bound; call via executor."""
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    rh, rw = smart_resize(img.height, img.width, spec.factor,
                          spec.min_pixels, spec.max_pixels)
    if (rw, rh) != img.size:
        img = img.resize((rw, rh), Image.BILINEAR)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), rw, rh


def build_payload(spec: ModelSpec, b64: str, instruction: str, temperature: float = 0.0):
    img_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    txt_part = {"type": "text", "text": instruction}
    content = [img_part, txt_part] if spec.img_first else [txt_part, img_part]
    msgs = []
    if spec.system:
        msgs.append({"role": "system", "content": spec.system})
    msgs.append({"role": "user", "content": content})
    payload = {"model": MODEL_NAME, "messages": msgs,
               "temperature": temperature, "max_tokens": spec.max_tokens}
    if spec.prefill:
        # force the assistant to continue from a fixed prefix (locks output
        # format for chatty Instruct models)
        msgs.append({"role": "assistant", "content": spec.prefill})
        payload["add_generation_prompt"] = False
        payload["continue_final_message"] = True
    return payload


async def wait_ready(ports=None, timeout=600):
    """Wait for the vLLM pool.

    - If ``ports`` is given, wait until *all* of those ports are healthy.
    - If ``ports`` is None, probe DEFAULT_PORTS and return as soon as *at least
      one* is healthy (typical when you only launched a subset of GPUs).
    """
    need_all = ports is not None
    ports = ports or DEFAULT_PORTS
    urls = health_urls(ports)
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        ready = set()
        while time.time() - t0 < timeout:
            for u in urls:
                if u in ready:
                    continue
                try:
                    async with s.get(u, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            ready.add(u)
                except Exception:
                    pass
            print(f"  ready {len(ready)}/{len(urls)} servers "
                  f"({int(time.time()-t0)}s)", flush=True)
            if need_all and len(ready) == len(urls):
                return True
            if (not need_all) and len(ready) >= 1:
                return True
            await asyncio.sleep(5)
    return len(ready) > 0


class GroundingPool:
    def __init__(self, spec: ModelSpec, ports: Optional[List[int]] = None,
                 concurrency_per: int = 14):
        self.spec = spec
        all_eps = endpoints(ports)
        healthy = _healthy_endpoints(all_eps)
        if healthy and len(healthy) < len(all_eps):
            dead = [e for e in all_eps if e not in healthy]
            print(f"[pool] {len(healthy)}/{len(all_eps)} endpoints healthy; "
                  f"excluding {dead}", flush=True)
        # fall back to all endpoints if the probe found none (e.g. slow startup)
        self.eps = healthy or all_eps
        self.sem = asyncio.Semaphore(len(self.eps) * concurrency_per)
        self._rr = 0

    async def _one(self, session, img, instruction, retries=3):
        spec = self.spec
        loop = asyncio.get_event_loop()
        try:
            b64, rw, rh = await loop.run_in_executor(_EXEC, prep_b64, spec, img)
        except Exception as e:
            return None, f"ERR:prep:{e}"
        async with self.sem:
            for attempt in range(retries):
                ep = self.eps[self._rr % len(self.eps)]
                self._rr += 1
                try:
                    payload = build_payload(spec, b64, instruction)
                    async with session.post(ep, json=payload,
                                            timeout=aiohttp.ClientTimeout(total=240)) as r:
                        data = await r.json()
                        txt = data["choices"][0]["message"]["content"] or ""
                        coord = spec.parse(spec.prefill + txt)
                        if coord is None:
                            norm = None
                        elif spec.coord_space == "norm1000":
                            norm = (coord[0] / 1000.0, coord[1] / 1000.0)
                        else:  # abs_resized
                            norm = (coord[0] / rw, coord[1] / rh)
                        return norm, txt
                except Exception as e:
                    if attempt == retries - 1:
                        return None, f"ERR:{e}"
                    await asyncio.sleep(1.0 + attempt)
        return None, "ERR:exhausted"

    async def ground_many(self, jobs: List[Tuple[Image.Image, str]]):
        """jobs: list of (PIL image, instruction). Returns list of
        (point_norm in [0,1] of the image content, raw_text)."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._one(session, img, ins) for img, ins in jobs]
            return await asyncio.gather(*tasks)

    async def _post(self, session, b64, rw, rh, instruction, temperature, retries=3):
        """One request against pre-encoded image; used for repeated sampling."""
        spec = self.spec
        async with self.sem:
            for attempt in range(retries):
                ep = self.eps[self._rr % len(self.eps)]
                self._rr += 1
                try:
                    payload = build_payload(spec, b64, instruction, temperature)
                    async with session.post(ep, json=payload,
                                            timeout=aiohttp.ClientTimeout(total=240)) as r:
                        data = await r.json()
                        txt = data["choices"][0]["message"]["content"] or ""
                        coord = spec.parse(spec.prefill + txt)
                        if coord is None:
                            norm = None
                        elif spec.coord_space == "norm1000":
                            norm = (coord[0] / 1000.0, coord[1] / 1000.0)
                        else:
                            norm = (coord[0] / rw, coord[1] / rh)
                        return norm, txt
                except Exception as e:
                    if attempt == retries - 1:
                        return None, f"ERR:{e}"
                    await asyncio.sleep(1.0 + attempt)
        return None, "ERR:exhausted"

    async def _post_ep(self, session, ep, b64, rw, rh, instruction,
                       temperature=0.0, retries=4):
        """First attempt hits the assigned endpoint (server affinity -> the group
        of instruction variants reuses that server's image prefix cache). On
        failure it FAILS OVER to the other endpoints, so a server that dies
        mid-run does not silently turn its share of requests into parse-fails."""
        spec = self.spec
        try:
            base_idx = self.eps.index(ep)
        except ValueError:
            base_idx = 0
        async with self.sem:
            for attempt in range(retries):
                cur = self.eps[(base_idx + attempt) % len(self.eps)]
                try:
                    payload = build_payload(spec, b64, instruction, temperature)
                    async with session.post(cur, json=payload,
                                            timeout=aiohttp.ClientTimeout(total=240)) as r:
                        if r.status != 200:
                            raise RuntimeError(f"http{r.status}")
                        data = await r.json()
                        txt = data["choices"][0]["message"]["content"] or ""
                        coord = spec.parse(spec.prefill + txt)
                        if coord is None:
                            norm = None
                        elif spec.coord_space == "norm1000":
                            norm = (coord[0] / 1000.0, coord[1] / 1000.0)
                        else:
                            norm = (coord[0] / rw, coord[1] / rh)
                        return norm, txt
                except Exception as e:
                    if attempt == retries - 1:
                        return None, f"ERR:{e}"
                    await asyncio.sleep(0.5 + attempt)
        return None, "ERR:exhausted"

    async def ground_grouped(self, groups, temperature: float = 0.0):
        """groups: list of (img_or_path, [instruction, ...]). Each image is
        encoded ONCE and all its instruction variants are routed to the SAME
        server (round-robin at the group level), so the shared image prefix is
        computed once and reused via vLLM prefix caching. Returns, per group, a
        list of (point_norm, raw) aligned to that group's instruction list."""
        loop = asyncio.get_event_loop()
        async with aiohttp.ClientSession() as session:
            enc = await asyncio.gather(*[
                loop.run_in_executor(_EXEC, prep_b64, self.spec, g[0])
                for g in groups])
            tasks, idx = [], []
            for gi, (img, instrs) in enumerate(groups):
                b64, rw, rh = enc[gi]
                ep = self.eps[gi % len(self.eps)]  # image -> fixed server
                for j, ins in enumerate(instrs):
                    tasks.append(self._post_ep(session, ep, b64, rw, rh, ins,
                                               temperature))
                    idx.append((gi, j))
            flat = await asyncio.gather(*tasks)
        out = [[None] * len(g[1]) for g in groups]
        for (gi, j), res in zip(idx, flat):
            out[gi][j] = res
        return out

    async def ground_repeated(self, jobs, k: int, temperature: float):
        """For each (img, instruction) job, draw k samples at `temperature`.
        Returns list (per job) of k (point_norm, raw) tuples. Image is encoded
        once per job (off the event loop), then k requests are fired and
        load-balanced across the pool."""
        loop = asyncio.get_event_loop()
        async with aiohttp.ClientSession() as session:
            enc = await asyncio.gather(*[
                loop.run_in_executor(_EXEC, prep_b64, self.spec, img)
                for img, _ in jobs])
            tasks, idx = [], []
            for i, (img, ins) in enumerate(jobs):
                b64, rw, rh = enc[i]
                for _ in range(k):
                    tasks.append(self._post(session, b64, rw, rh, ins, temperature))
                    idx.append(i)
            flat = await asyncio.gather(*tasks)
        per = [[] for _ in jobs]
        for i, res in zip(idx, flat):
            per[i].append(res)
        return per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--dataset", default="screenspot_v2")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    if args.wait:
        ok = asyncio.run(wait_ready(args.ports))
        print("READY" if ok else "TIMEOUT")
    if args.smoke:
        from grounding import load_items, hit
        spec = get_spec(args.model)
        items = load_items(args.dataset, limit=args.n)
        imgs = [(Image.open(it.img_path).convert("RGB"), it.instruction) for it in items]
        pool = GroundingPool(spec, args.ports)
        res = asyncio.run(pool.ground_many(imgs))
        n_ok = 0
        for it, (pt, raw) in zip(items, res):
            if pt is None:
                print(it.id, "PARSE-FAIL |", raw[-70:].replace("\n", " ")); continue
            px = pt[0] * it.img_w, pt[1] * it.img_h
            h = hit(it.bbox, px)
            n_ok += int(h)
            print(f"{it.id[:26]:26s} hit={h} pt=({px[0]:.0f},{px[1]:.0f}) "
                  f"norm=({pt[0]:.3f},{pt[1]:.3f})")
        print(f"[{args.model}] smoke acc {n_ok}/{len(items)}")
