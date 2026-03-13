import requests

def debug_sst(url):
    print("\n" + "="*50)
    print(f"Checking: {url}")
    print("="*50)
    r = requests.get(url)
    text = r.text.strip()
    lines = text.split('\n')
    print(f"Total lines: {len(lines)}")
    from collections import Counter
    lengths = [len(l.replace(" ", "").replace("\r", "")) for l in lines]
    print(f"Lengths counter (Sorted): {sorted(Counter(lengths).items())}")

if __name__ == "__main__":
    debug_sst('https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/2026/him_sst_pac_D20260101.txt')
    debug_sst('https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/2025/him_sst_pac_D20251201.txt')
