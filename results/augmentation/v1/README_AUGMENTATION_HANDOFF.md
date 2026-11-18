# BÁO CÁO BÀN GIAO KỸ THUẬT: MODULE DATA AUGMENTATION (CASEKG v1.1)
**Dự án:** CASEKG – Chẩn đoán Bệnh nguyên Đau thắt lưng (Low Back Pain) bằng Đồ thị Tri thức & NLP  
**Người thực hiện:** Phạm Hoàng Minh Hiếu
**Ngày bàn giao:** 18/11/2026  
**Phiên bản phát hành:** `v1.1` (Cleaned Ontology & Isolated Train-Only Pool)  
**Đầu mối tiếp nhận:**  Phúc (Clinical Audit) & Nhóm trưởng / Minh (Model Training & Metrics)

---

## PHẦN 1: TỔNG QUAN & PHẠM VI KỸ THUẬT ĐÃ HOÀN THÀNH (100% SCOPE)

Minh Hiếu đã hoàn tất 100% các mục tiêu kỹ thuật được giao trong Milestone 1 theo đúng bản phân công công việc:

### 1. Chuẩn hóa & Làm sạch Toàn diện Workspace
- Hợp nhất thành công các thư mục bị phân mảnh (`data/`, `resources/`, `models/`, `results/`, `src/`) về trực tiếp gốc workspace mà **không làm thay đổi bất kỳ byte nào** của mã nguồn gốc.
- Đã kiểm chứng qua công cụ `tools/maintenance/seal_code_layout.py`: **479/479 tệp gốc đạt 100% khớp chữ ký SHA256**.
- Xử lý triệt để lỗi xung đột nạp C-extension (`_duckdb`, `scipy.sparse`, `numpy`) trước PyTorch trên Windows (khắc phục lỗi cạn kiệt 128 khe Fiber Local Storage - FLS slots). Toàn bộ **455/455 unit tests gốc của hệ thống chạy thông suốt 100%**.

### 2. Bảng Ánh xạ Tọa độ Mệnh đề Toàn văn bản (`clause_offset_table.json`)
- Tái tạo thành công bảng ánh xạ tọa độ `[clause_start, clause_end]` từ cấp mệnh đề (`clause_uid`) quy đổi về vị trí chính xác trên toàn văn bản bệnh án `prediagnosis_text`.
- **Độ chính xác:** Đạt **100,0% (0 lỗi lệch offset)** trên toàn bộ **3.708 thực thể** của cohort 57 ca hiện hành và **23.288 thực thể** của toàn bộ 401 ca trong kho dữ liệu.
- Vị trí lưu trữ: `resources/augmentation/clause_offset_table.json`.

### 3. Từ điển Đồng nghĩa Chuẩn Khái niệm Lâm sàng (`lbp_synonym_dictionary.json` v2.0)
- Xóa bỏ triệt để từ điển thủ công và cơ chế gộp chéo đa mã dễ gây rò rỉ.
- Xây dựng từ điển đồng nghĩa chuẩn đơn trị 1-1 theo mã CURIE từ 4 ontology chuẩn: **HP** (Phenotype), **SYMP** (Symptom), **UBERON** (Anatomy), **MAXO** (Medical Action):
  - **1.110 khái niệm** y sinh chuẩn.
  - **4.702 lookup keys** đã được khử nhập nhằng từ vựng.
  - **0 mã bệnh `MONDO`** (Triệt tiêu 100% nguy cơ biến triệu chứng thành tên bệnh).
  - Khóa bảo vệ nghiêm ngặt **674 chuỗi chẩn đoán đích** và **74 token bệnh nguyên lâm sàng**.
  - Áp dụng bộ lọc runtime loại bỏ hoàn toàn các mã kỹ thuật (CDISC codes) và danh pháp giải phẫu lý thuyết học thuật cứng nhắc (v1.1: loại bỏ `axial skeleton`, `initial segment`, `zeugopod`, `posterior abdominal wall`, `lumbar skeleton`...).
- Vị trí lưu trữ: `resources/augmentation/lbp_synonym_dictionary.json`.

### 4. Package Mã nguồn `src/casekg/augmentation/` & Bộ Unit Test Chuẩn mực
- Đóng gói hoàn chỉnh thành một module sạch, có type hints, docstring đầy đủ:
  - `types.py`: Định nghĩa các cấu trúc dữ liệu bất biến (`ClauseSpan`, `EntitySpan`, `HydratedDocument`, `ReplacementEdit`, `AugmentedRecord`).
  - `hydrator.py` (`SpanHydrator`): Khôi phục tọa độ thực thể toàn văn bản từ nhiều nguồn đặc trưng, tự động khóa bảo vệ các thực thể có dấu phủ định, chỉ số xét nghiệm, dấu sinh tồn hoặc nằm trong danh sách đen chẩn đoán.
  - `synonym_swapper.py` (`SynonymSwapper`): Thực hiện hoán đổi từ đồng nghĩa 1-1 theo mã CURIE kèm thuật toán **Delta Offset Tracking** (bảo toàn offset 100%, Zero-Drift tuyệt đối).
  - `pipeline.py` (`AugmentationPipeline`): Quản lý luồng xử lý xác định (deterministic qua `random.Random(seed)`), kiểm tra tính bất biến và cung cấp hàm lọc dữ liệu an toàn `select_fold_variants()`.
- **Bộ kiểm thử tự động:** `tests/unit/test_casekg_augmentation.py` vượt qua **6/6 test cases** trong 0.69 giây:
  1. `test_clause_offset_table_integrity`: Kiểm chứng bảng offset mệnh đề.
  2. `test_span_hydrator_alignment`: Kiểm chứng khôi phục tọa độ thực thể.
  3. `test_synonym_dictionary_no_leakage`: Kiểm chứng 0% rò rỉ nhãn bệnh.
  4. `test_synonym_swapper_zero_drift`: Kiểm chứng độ lệch offset bằng 0 sau hoán đổi.
  5. `test_augmentation_pipeline_determinism`: Kiểm chứng tính lặp lại tuyệt đối với cùng một seed.
  6. `test_pipeline_rare_multipliers_and_isolation`: Kiểm chứng hệ số nhân lớp hiếm và khóa cách ly fold.

### 5. Gói Dữ liệu Xuất bản Chuẩn hóa (`results/augmentation/v1/`)
- Xuất bản thành công tập dữ liệu tăng cường phiên bản v1.1:
  - `TRAIN_AUGMENTED_VARIANTS.jsonl`: **59 biến thể thuần tập `train`** (loại bỏ hoàn toàn 19 biến thể thuộc tập `dev` để ngăn rò rỉ dữ liệu đánh giá).
  - `AUGMENTED_VARIANTS.jsonl`: Toàn bộ 78 biến thể của cohort 57 ca.
  - `CLINICAL_AUDIT_40_SAMPLES.md` & `.jsonl`: **40 mẫu đối chiếu lâm sàng** phục vụ quy trình audit của Phúc.
  - `MANIFEST.json`: Bảng băm SHA256 và siêu dữ liệu kiểm chứng toàn vẹn.

---

## PHẦN 2: ĐỐI SOÁT TRỰC DIỆN — HIẾU ĐÃ THAY ĐỔI GÌ SO VỚI TRƯỚC ĐÓ?

| Trục Tiêu chí | Thực trạng TRƯỚC ĐÓ (Baseline thô) | Thực trạng HIỆN TẠI (Sau khi Hiếu hoàn tất v1.1) | Giá trị Kỹ thuật Đạt được |
| :--- | :--- | :--- | :--- |
| **Tính toàn vẹn mã nguồn & Layout** | Codebase bị phân mảnh trong `CASEKG_SOURCE`, tồn tại symlink/junction và file cấu hình tạm. | Gốc workspace sạch sẽ, đồng bộ; **479/479 tệp mã nguồn gốc khớp SHA256 100%**. Toàn bộ 455 test gốc pass. | Hệ thống ổn định tuyệt đối, không có rủi ro phụ thuộc ẩn. |
| **Độ chính xác Tọa độ Thực thể** | Tọa độ thực thể chỉ có ở cấp mệnh đề tách rời (`clause_uid`). Cố gắng hoán đổi trực tiếp sẽ làm gãy cấu trúc ngữ pháp và lệch toàn bộ offset của văn bản sau đó. | **100,0% thực thể** được ánh xạ chính xác về văn bản toàn phần `prediagnosis_text`. Thuật toán Delta Offset Tracking cập nhật tọa độ tự động sau mỗi lần thay thế. | **Zero Offset Drift:** Đồ thị tri thức rút trích sau này được bảo tồn chính xác từng nút và cạnh. |
| **An toàn Chống Rò rỉ Nhãn (Label Leakage)** | Từ điển thô gộp chung mã `MONDO` khiến triệu chứng có thể bị hoán đổi thành chẩn đoán cuối cùng; chứa từ rác kỹ thuật (`vstest`). | Phân tách đơn trị CURIE, loại bỏ hoàn toàn `MONDO`, chặn 674 tên bệnh đích; lọc sạch thuật ngữ giải phẫu sách vở cứng nhắc (v1.1). | **Zero Label Leakage:** Mô hình bị triệt tiêu khả năng "học lỏm" tên bệnh trong văn bản đầu vào. |
| **Mật độ Mẫu Train Lớp Hiếm (2-Fold CV trên 57 ca)** | Khi chia 2 fold, nhãn cực hiếm `REFERRED_PAIN` **chỉ có duy nhất 1 ca train** ($n=1$). Mô hình bị "đói gradient" nghiêm trọng, Macro-F1 lịch sử = 0.0. | Áp dụng hệ số nhân tầng hiếm: `REFERRED_PAIN` tăng lên **5 mẫu/fold train**; `INFLAMMATORY` và `METABOLIC` tăng lên **8–11 mẫu/fold train**. | Cung cấp đủ mật độ biểu diễn vector để bộ phân loại tìm được siêu phẳng (hyperplane) phân lớp. |
| **Cách ly Phân vùng Train - Dev** | Nguy cơ dùng nhầm các ca thuộc tập `dev` để sinh tăng cường rồi nạp vào tập `train`. | Tách riêng tệp `TRAIN_AUGMENTED_VARIANTS.jsonl` (chỉ gồm các ca gốc `split == 'train'`). Khóa `cv_isolation_key` theo `lineage_group_id`. | Triệt tiêu 100% rủi ro rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm định. |

---

## PHẦN 3: LỘ TRÌNH BÀN GIAO & HƯỚNG DẪN CHI TIẾT CHO THÀNH VIÊN TIẾP THEO

### 1. DÀNH CHO PHÚC (Phụ trách Thẩm định Y khoa & Clinical Sample Audit)

#### A. Hiếu đã chuẩn bị gì cho Phúc?
- **Tệp dữ liệu thẩm định:** [CLINICAL_AUDIT_40_SAMPLES.md](file:///c:/Users/LEGION/Desktop/BKUni/NCKH/results/augmentation/v1/CLINICAL_AUDIT_40_SAMPLES.md) (và bản JSONL tương ứng: `results/augmentation/v1/CLINICAL_AUDIT_40_SAMPLES.jsonl`).
- **Nội dung:** Gồm đúng 40 cặp mẫu đối chiếu lâm sàng, phân tầng tập trung vào các lớp bệnh hiếm (`REFERRED_PAIN`, `INFLAMMATORY`, `METABOLIC`) và các lớp thường gặp. Mỗi mẫu đều hiển thị:
  - Mã ca (`parent_document_id`).
  - Nhãn nhóm bệnh (`disease_family_label`).
  - Danh sách cụm từ thay thế theo chuẩn CURIE kèm tọa độ mới.
  - Đoạn văn bệnh án gốc vs. Đoạn văn bệnh án sau khi tăng cường.
  - Xác nhận khớp offset 100% bằng máy.

#### B. Việc Phúc cần làm tiếp theo (Hạn chót: 22/11/2026):
Phúc lướt qua 40 mẫu và thẩm định theo 2 tiêu chí lâm sàng chính:
1. **Tính tự nhiên & Ngữ pháp (Grammar & Fluency):** Câu văn tiếng Anh sau khi thay thế từ đồng nghĩa y khoa đọc có tự nhiên, đúng văn phong bệnh án không?
2. **Bảo toàn Ngữ nghĩa Lâm sàng (Clinical Semantic Preservation):** Từ thay thế có giữ nguyên bản chất của triệu chứng/khám nghiệm không? Có làm thay đổi hướng chẩn đoán của bác sĩ không?
3. **Đầu ra mong đợi:**
   - Đánh dấu **Pass / Fail** cho từng mẫu (hoặc xác nhận toàn bộ nếu đạt yêu cầu).
   - Trả về **Tỷ lệ đạt chuẩn lâm sàng (Clinical Acceptance Rate, kỳ vọng $\ge 95\%$)** để nhóm đưa con số này vào bảng đánh giá chất lượng dữ liệu trong bài báo khoa học.
   - Nếu có từ nào Phúc thấy không chuẩn y khoa, chỉ cần ghi chú lại cụm từ đó để nhóm đưa vào danh sách chặn vĩnh viễn.

---

### 2. DÀNH CHO MINH (Nhóm trưởng — Phụ trách Huấn luyện Mô hình & Đánh giá Metric)

#### A. Hiếu đã chuẩn bị gì cho Minh?
- **Gói dữ liệu biến thể thuần Train:** [TRAIN_AUGMENTED_VARIANTS.jsonl](file:///c:/Users/LEGION/Desktop/BKUni/NCKH/results/augmentation/v1/TRAIN_AUGMENTED_VARIANTS.jsonl) (59 biến thể chuẩn hóa, đã lọc sạch các ca `dev`).
- **Hàm hook chống rò rỉ fold:** `select_fold_variants()` trong `src/casekg/augmentation/pipeline.py`.

#### B. Việc Minh cần làm tiếp theo:

##### B1. Tích hợp Hook 3 dòng vào Pipeline Huấn luyện (`ObservationEvidenceDiagnosis`)
Trong vòng lặp chia fold (ví dụ `StratifiedGroupKFold` 2-fold), Minh chèn đoạn code sau:

```python
import json
from casekg.augmentation import select_fold_variants

# 1. Đọc sẵn kho biến thể đã chuẩn hóa của Hiếu
with open("results/augmentation/v1/TRAIN_AUGMENTED_VARIANTS.jsonl", "r", encoding="utf-8") as f:
    variants_pool = [json.loads(line) for line in f if line.strip()]

# 2. Trong mỗi fold huấn luyện:
for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter.split(cohort_records, ...)):
    train_parent_ids = {cohort_records[i]["document_id"] for i in train_indices}
    val_parent_ids = {cohort_records[i]["document_id"] for i in val_indices}
    
    # Lọc biến thể: CHỈ lấy các biến thể sinh từ train_parent_ids (cách ly 100% khỏi val)
    train_variants = select_fold_variants(variants_pool, train_fold_parent_ids=train_parent_ids)
    
    # Gộp vào tập train hiện tại để huấn luyện
    fold_train_data = [cohort_records[i] for i in train_indices] + train_variants
    fold_val_data = [cohort_records[i] for i in val_indices]  # Tập val GIỮ NGUYÊN BẢN 100%
    
    # 3. Fit mô hình trên fold_train_data và evaluate trên fold_val_data
    model.fit(fold_train_data)
    predictions = model.predict(fold_val_data)
```

##### B2. Chạy Thực nghiệm Đối chiếu 3 Mốc (Ablation Study)
Minh chạy thực nghiệm đánh giá hiệu năng trên 3 cấu hình thử nghiệm sau:
1. **Mốc 1 (Baseline):** Huấn luyện thuần trên 57 ca chính gốc.
2. **Mốc 2 (+ Data Augmentation):** Huấn luyện trên 57 ca gốc + 59 biến thể từ `TRAIN_AUGMENTED_VARIANTS.jsonl`.
3. **Mốc 3 (+ Augmentation + Weighted Auxiliary):** Gộp thêm **379 ca** từ `data/isolated_development/orbit_auxiliary/diagnosis_auxiliary_train.jsonl` (trong đó có 28 ca Đau quy chiếu và 22 ca Viêm) với trọng số mẫu thấp (`sample_weight = 0.25`).
   - *Mục đích của Mốc 3:* Giải quyết dứt điểm giới hạn toán học $N_{test} = 1$ ở tập kiểm định của lớp hiếm, giúp đường biên quyết định của mô hình ổn định nhất.

##### B3. Xuất Báo cáo Kết quả
Minh xuất bảng chỉ số đánh giá:
- Overall Micro-F1 & Accuracy.
- Macro-F1 toàn diện và **Rare-Class Macro-F1 ($K_{rare}$)** cho 3 nhóm: `REFERRED_PAIN`, `INFLAMMATORY`, `METABOLIC`.
- Khoảng tin cậy **95% Bootstrap Confidence Interval** và kiểm định **Wilcoxon signed-rank test** để Hiếu và nhóm đưa vào Section 5 của bài báo.

---

## PHẦN 4: TRẠNG THÁI CHỜ & VAI TRÒ KẾ TIẾP CỦA MINH HIẾU

Sau khi hoàn tất bàn giao gói kỹ thuật `v1.1`, Minh Hiếu sẽ:

1. **Hỗ trợ Kỹ thuật & Tích hợp:**
   - Phối hợp với Minh trong quá trình tích hợp vào `ObservationEvidenceDiagnosis`, hỗ trợ tinh chỉnh tham số `swap_prob` hoặc trọng số mẫu `sample_weight` nếu mô hình cần cân chỉnh độ nhạy.
2. **Soạn thảo Bài báo Khoa học (Template LNCS):**
   - Viết hoàn thiện bản thảo **Section 2.3 (Data Preprocessing & Augmentation Strategy)**: Trình bày chi tiết cơ chế *Ontology-guided Single-CURIE Synonym Swapping*, thuật toán *Delta Offset Tracking*, và giao thức *GroupKFold Train-Only Isolation*.
   - Phối hợp cùng Phúc dự thảo phần **Abstract** (150–250 từ) và lựa chọn **Keywords**.
3. **Tổng hợp Kết quả:**
   - Sau khi Phúc hoàn tất review lâm sàng và Minh chạy xong Ablation Study, Hiếu sẽ tổng hợp số liệu để viết phần phân tích thực nghiệm và vẽ biểu đồ hiệu năng.

---
*Tài liệu được lập và xác thực tự động bởi Minh Hiếu — Nhóm Nghiên cứu CASEKG.*  
*Toàn bộ chữ ký SHA256 được lưu trữ tại `results/augmentation/v1/MANIFEST.json`.*
