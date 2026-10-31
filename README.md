# wachirawit.chai — Model A: DCS two-layer flow phantom simulation

จำลองกราฟ autocorrelation g2(τ) ของ DCS ด้วย Monte Carlo (MCX) สำหรับ phantom สองชั้นของ
Nakabayashi et al. (2023), *Biomed. Opt. Express* 14(10) 5358 แล้วเทียบกับข้อมูลวัดจริงของเปเปอร์

ภาพรวมการทำงานมี 3 ขั้น:

```
1. run-mcx    ยิงโฟตอนใน phantom (ทำครั้งเดียว)    -> data/pathlengths.npz
2. calibrate  หา a, D_B0 ด้วย Nelder-Mead + วัดผล     -> final_result/
3. generate   สร้าง g2 จาก flow ที่กำหนดเอง          -> generated/g2_generated.csv
```

**ถ้าต้องการแค่สร้าง g2 ให้ข้ามไปขั้นที่ 3 ได้เลย** เพราะใน repo มีผลของขั้นที่ 1 และ 2 อยู่แล้ว
(`data/pathlengths.npz` และ `final_result/model_a.json`)

---

## 0. ติดตั้ง

### สิ่งที่ต้องมี
- Mac Apple Silicon (M1–M4), Homebrew, Python 3.11 (`brew install python@3.11`)
- ข้อมูลวัดจริงอยู่ที่ `../core/data/` (โฟลเดอร์ข้างๆ repo นี้) ต้องมี `metadata.csv` และ
  `g2_SDS1cm.csv`, `g2_SDS2cm.csv`, `g2_SDS3cm.csv` ใช้ในขั้น `calibrate` เท่านั้น

```
project/
├── core/data/          ข้อมูลวัดจริง
└── wachirawit.chai/    repo นี้
```

### ติดตั้ง environment

```bash
git clone https://github.com/Senior-ProjectMCX/wachirawit.chai.git
cd wachirawit.chai
bash scripts/setup_env.sh        # สร้าง .venv, ติดตั้ง numpy/scipy/pandas/matplotlib, build pmcxcl
source .venv/bin/activate
```

`setup_env.sh` build `pmcxcl` (MCX สำหรับ OpenCL) จาก source เพราะ wheel บน PyPI เป็น x86_64 อย่างเดียว
ใช้บน Apple Silicon ไม่ได้ ถ้าจะใช้แค่ `generate` ไม่ต้องมี `pmcxcl` ก็ได้
(`pip install -r requirements.txt` พอ)

ทุกคำสั่งด้านล่างรันจากโฟลเดอร์ `wachirawit.chai/` โดยเปิด `.venv` ไว้แล้ว

---

## 1. `run-mcx`: จำลองโฟตอน (ทำครั้งเดียว)

```bash
python src/main.py run-mcx                     # 1e8 โฟตอน, ~45 วินาทีบน M4
python src/main.py run-mcx --nphoton 2e8       # เปลี่ยนจำนวนโฟตอน
```

- ผลลัพธ์คือ `data/pathlengths.npz`: ระยะทาง (L) และ momentum transfer (Y) ของทุกโฟตอนที่ถึง detector
  แยกตามชั้นและ SDS (`det_{1,2,3}_{Ls,Ld,Ys,Yd}`)
- ผลนี้**ไม่ขึ้นกับ flow** ใช้ซ้ำได้กับทุก flow
- ต้องรันใหม่เมื่อเปลี่ยน geometry หรือ optics ใน `config/parameters.py` เท่านั้น
  (μa, μs′, g, ความหนาชั้น, ตำแหน่ง/รัศมี detector)

## 2. `calibrate`: หาค่าสมการ flow → αD_B และวัดผล

```bash
python src/main.py calibrate                   # ~5 นาที
```

ทำ 3 อย่าง:
1. หา `a`, `D_B0` ใน αD_B = a · max(FR, 0) + D_B0 โดยให้ g2 จำลองใกล้ g2 จริงที่สุด
   (least squares, Nelder-Mead) บน test set 660 trials (`dataset/testset_closest_k5.csv`)
2. คำนวณ metric: R² median/mean, pooled RMSE, BFI-X ratio, BFI-Y ratio (พิมพ์ออกหน้าจอ)
3. วาดกราฟทุก 132 เงื่อนไข (จริงเส้นดำ, Model A เส้นปะ)

ผลลัพธ์ใน `final_result/`:

| ไฟล์ | เนื้อหา |
| --- | --- |
| `model_a.json` | a, D_B0, β, metric ทั้งหมด (ขั้น `generate` อ่านไฟล์นี้) |
| `conditions/cond_<ตื้น>_<ลึก>.png` | กราฟ 1 เงื่อนไข, SDS 1/2/3 cm |
| `all_conditions.pdf` | ทุกเงื่อนไขรวมไฟล์เดียว |
| `grid_SDS{1,2,3}cm.png` | 132 เงื่อนไขในภาพเดียวต่อ SDS |
| `metrics_per_condition.csv` | R², RMSE ต่อเงื่อนไข |

ตัวเลือก: `--pathlengths <ไฟล์ npz>`, `--testset <ไฟล์ csv>`, `--out <โฟลเดอร์>`

## 3. `generate`: สร้าง g2 จาก flow ที่กำหนด

```bash
python src/main.py generate --flows 5,20 12,30 0,40           # (ชั้นตื้น, ชั้นลึก) ml/min/100g
python src/main.py generate --flow-csv my_flows.csv           # อ่าน flow จากไฟล์
python src/main.py generate --flows 5,20 --out results/a.csv  # เลือกที่เก็บไฟล์
```

- `my_flows.csv` ต้องมีคอลัมน์ `FR_shallow, FR_deep`
- ผลลัพธ์ (ค่าเริ่มต้น `generated/g2_generated.csv`) มี 1 แถวต่อ 1 flow ต่อ 1 SDS:

  | FR_shallow | FR_deep | SDS | tau_0us | tau_1us | … | tau_500us |
  | --- | --- | --- | --- | --- | --- | --- |
  | 5 | 20 | 1cm | 1.405 | … | … | … |

- เร็วมาก (ไม่ถึง 1 วินาทีต่อหลาย flow) ไม่ต้องรัน MCX และไม่ต้องใช้ข้อมูลจริง

### เรียกจาก Python

```python
import sys; sys.path.insert(0, "src")
from generate import generate_g2

g2 = generate_g2([(5, 20), (12, 30)])      # shape (จำนวน flow, 3 SDS, 501 τ), τ = 0..500 µs
g2_1cm = g2[0, 0]                          # flow (5, 20) ที่ SDS 1 cm

g2 = generate_g2([(5, 20)], a=1e-7, db0=5e-7, beta=0.5)   # กำหนดค่าเองแทนค่าที่ fit ไว้
```

---

## โครงสร้าง repo

| ไฟล์ | หน้าที่ |
| --- | --- |
| `config/parameters.py` | ค่าคงที่ทั้งหมด: optics, geometry, จำนวนโฟตอน, สมการ flow → αD_B |
| `src/main.py` | คำสั่ง `run-mcx`, `calibrate`, `generate`, `all` |
| `src/mcx_engine.py` | สร้าง voxel phantom และรัน MCX |
| `src/dcs_physics.py` | g1/g2 จากโฟตอน (Wu et al. 2021 Eq. 5 + Siegert) |
| `src/calibrate.py` | Nelder-Mead หา a, D_B0 และคำนวณ metric |
| `src/bfi_fit.py` | fit BFI แบบ semi-infinite (เปเปอร์ Eq. 3) สำหรับ BFI-X/Y |
| `src/plot_results.py` | กราฟจริง vs Model A |
| `src/generate.py` | สร้าง g2 จาก flow ที่กำหนด |
| `scripts/make_testset.py` | สร้าง test set: `--k 5` = 5 trials ต่อเงื่อนไขที่ flow วัดได้ใกล้ค่าตั้งที่สุด |
| `scripts/setup_env.sh` | ติดตั้ง environment |
| `data/pathlengths.npz` | ผล MCX 1e8 โฟตอน |
| `dataset/` | ตาราง flow (`Dataset_metadata.csv`) และ test set |

## รายละเอียดโมเดล

| ส่วน | ค่า |
| --- | --- |
| Geometry | กล่อง 100 × 50 × 40.5 mm: ชั้นตื้น 5.5 mm + ชั้นลึก 35 mm (เปเปอร์ Sec. 2.1), ทุกผิวเป็น Fresnel, voxel 0.5 mm |
| Source / detector | pencil beam ที่ x = 35 mm, detector รัศมี 1 mm ที่ SDS 1/2/3 cm |
| Optics (785 nm) | μa 0.056 mm⁻¹, μs′ 0.481 mm⁻¹, g 0.89, n 1.37 (ทั้งสองชั้น) |
| Flow → αD_B | αD_B = a · max(FR, 0) + D_B0, ใช้ a เดียวกันทั้งสองชั้น |
| g2 | g1 แบบ momentum transfer, g2 = 1 + β·g1², β ต่อ SDS = ค่าเฉลี่ย g2(0) จริง − 1 |

## ผลลัพธ์ปัจจุบัน

a = 9.425e-8 mm²/s ต่อ ml/min/100g, D_B0 = 6.675e-7 mm²/s, β = 0.405 / 0.494 / 0.485
(fit บน 660 trials, 5,000 โฟตอนต่อ SDS)

| SDS | 1 cm | 2 cm | 3 cm |
| --- | --- | --- | --- |
| R² median (ต่อ trial) | 0.942 | 0.968 | 0.925 |
| R² mean | 0.825 | 0.937 | 0.883 |
| Pooled RMSE ของ g2 | 0.0332 | 0.0225 | 0.0315 |
| BFI-X ratio median / mean | 0.74 / 0.75 | 1.07 / 1.04 | 1.21 / 1.24 |
| BFI-Y ratio median / mean | 1.41 / 1.43 | 0.94 / 0.92 | 0.62 / 0.64 |

- BFI-X = BFI จาก fitter ของเรา (เปเปอร์ Eq. 3) เทียบ g2 จำลองกับ g2 จริง
- BFI-Y = เทียบกับคอลัมน์ BFI ใน metadata (อัลกอริทึมของแล็บ ซึ่งเราไม่รู้วิธีคำนวณ)
- งานทดลองก่อนหน้า (geometry อื่น, เทียบ Model B, ทดลองจำนวนโฟตอนและ K) ดูได้ใน git history:
  branch `k5-photon-test`, commit `a7d4638`
