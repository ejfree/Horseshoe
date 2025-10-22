from PIL import Image, ImageFilter

def _sample_bg_color(img: Image.Image, patch: int = 64):
    w, h = img.size
    areas = [
        (0, 0, min(patch, w), min(patch, h)),
        (max(0, w - patch), 0, w, min(patch, h)),
        (0, max(0, h - patch), min(patch, w), h),
        (max(0, w - patch), max(0, h - patch), w, h),
    ]
    rs, gs, bs = [], [], []
    for box in areas:
        region = img.crop(box).convert("RGB")
        for r, g, b in region.getdata():
            rs.append(r); gs.append(g); bs.append(b)
    rs.sort(); gs.sort(); bs.sort()
    mid = len(rs) // 2
    return rs[mid], gs[mid], bs[mid]


def _largest_component(mask_img: Image.Image) -> Image.Image:
    w, h = mask_img.size
    m = mask_img.load()
    visited = [[False]*w for _ in range(h)]
    best_area = 0
    best_label_pixels = []
    from collections import deque
    for y in range(h):
        for x in range(w):
            if m[x, y] < 128 or visited[y][x]:
                continue
            q = deque([(x, y)])
            visited[y][x] = True
            pixels = [(x, y)]
            while q:
                cx, cy = q.popleft()
                for nx, ny in ((cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)):
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and m[nx, ny] >= 128:
                        visited[ny][nx] = True
                        q.append((nx, ny))
                        pixels.append((nx, ny))
            if len(pixels) > best_area:
                best_area = len(pixels)
                best_label_pixels = pixels
    out = Image.new('L', (w, h), 0)
    out_px = out.load()
    for x, y in best_label_pixels:
        out_px[x, y] = 255
    return out


def remove_yellow(src_path: str, dst_path: str,
                  bg_threshold: int = 60,
                  contract: int = 0,
                  expand: int = 1,
                  edge_clear: int = 0,
                  defringe_passes: int = 2) -> None:
    """Remove parchment-yellow area by color distance to the background.

    - Estimates background color from image corners.
    - Pixels within `bg_threshold` (RGB distance) become transparent.
    - Optional `contract` shrinks the kept area to avoid halos.
    """
    im = Image.open(src_path).convert("RGBA")
    rgb = im.convert("RGB")
    br, bg, bb = _sample_bg_color(rgb)

    w, h = im.size
    alpha = Image.new("L", (w, h), 255)
    a_px = alpha.load()
    r_px = rgb.load()

    thr2 = bg_threshold * bg_threshold
    for y in range(h):
        for x in range(w):
            r, g, b = r_px[x, y]
            dr = r - br; dg = g - bg; db = b - bb
            if (dr*dr + dg*dg + db*db) <= thr2:
                a_px[x, y] = 0

    if contract and contract > 0:
        size = contract if contract % 2 == 1 else contract + 1
        alpha = alpha.filter(ImageFilter.MinFilter(size=size))
    if expand and expand > 0:
        size = expand if expand % 2 == 1 else expand + 1
        alpha = alpha.filter(ImageFilter.MaxFilter(size=size))

    # Keep only the largest connected component to remove stray border dots
    alpha = _largest_component(alpha)

    # Optionally clear a small margin near edges
    if edge_clear and edge_clear > 0:
        w, h = alpha.size
        a = alpha.load()
        ec = edge_clear
        for y in range(h):
            for x in range(w):
                if x < ec or y < ec or x >= w-ec or y >= h-ec:
                    a[x, y] = 0

    # Remove yellow fringe near transparent areas while preserving interior
    if defringe_passes and defringe_passes > 0:
        hsv = rgb.convert("HSV")
        H, S, V = hsv.split()
        H_px, S_px, V_px = H.load(), S.load(), V.load()
        a = alpha.load()
        w, h = alpha.size
        for _ in range(defringe_passes):
            to_zero = []
            for y in range(h):
                for x in range(w):
                    if a[x, y] < 128:
                        continue
                    hv = H_px[x, y] * 360.0 / 255.0
                    sv = S_px[x, y]
                    vv = V_px[x, y]
                    # Yellow band detection
                    if 20 <= hv <= 65 and sv >= 40 and vv >= 140:
                        # near edge? any neighbor transparent
                        edge = False
                        for nx in (x-1, x, x+1):
                            for ny in (y-1, y, y+1):
                                if 0 <= nx < w and 0 <= ny < h and (nx != x or ny != y):
                                    if a[nx, ny] < 64:
                                        edge = True
                                        break
                            if edge:
                                break
                        if edge:
                            to_zero.append((x, y))
            for x, y in to_zero:
                a[x, y] = 0

    r, g, b, _ = im.split()
    out = Image.merge("RGBA", (r, g, b, alpha))
    out.save(dst_path)


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser(description="Remove parchment background by color distance")
    p.add_argument("src", help="source PNG path")
    p.add_argument("dst", help="output PNG path")
    p.add_argument("--bg-threshold", type=int, default=60, help="RGB distance threshold to background")
    p.add_argument("--contract", type=int, default=0, help="contract mask to avoid halo (odd px)")
    p.add_argument("--expand", type=int, default=1, help="expand mask to include outline (odd px)")
    p.add_argument("--edge-clear", type=int, default=0, help="clear this many pixels at edges")
    p.add_argument("--defringe-passes", type=int, default=2, help="remove yellow fringe near edges (iterations)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    remove_yellow(args.src, args.dst, args.bg_threshold, args.contract, args.expand, args.edge_clear, args.defringe_passes)
