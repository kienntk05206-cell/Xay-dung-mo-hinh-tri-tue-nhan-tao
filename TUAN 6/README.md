# BÀI 6: TRANSFER LEARNING – FGVC-AIRCRAFT

## 1. Giới thiệu
Bài thực hành xây dựng mô hình phân loại hình ảnh máy bay sử dụng kỹ thuật Transfer Learning với mô hình EfficientNet-B0 pretrained trên ImageNet.

## 2. Bộ dữ liệu
- Tên bộ dữ liệu: FGVC-Aircraft.
- Số lớp phân loại: 100 loại máy bay.
- Tập Train: 3.334 ảnh.
- Tập Validation: 3.333 ảnh.
- Tập Test: 3.333 ảnh.

Nguồn dữ liệu: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

## 3. Mô hình và phương pháp
- Mô hình: EfficientNet-B0.
- Trọng số pretrained: ImageNet.
- Kích thước ảnh đầu vào: 224 × 224.
- Hàm mất mát: CrossEntropyLoss.
- Bộ tối ưu: AdamW.
- Learning Rate: 0.0001.
- Số Epoch huấn luyện: 5.
- Môi trường thực nghiệm: Google Colab, GPU Tesla T4.

## 4. Kết quả thực nghiệm

| Chỉ số | Kết quả |
|---|---:|
| Best Validation Accuracy | 39.81% |
| Test Accuracy | 41.82% |
| Test Loss | 2.3570 |

## 5. Cấu trúc dự án

- `Bai_6_Transfer_Learning_Nguyen_Trung_Kien.ipynb`: Notebook chứa mã nguồn và kết quả thực nghiệm.
- `README.md`: Giới thiệu dự án và kết quả.

## 6. Cách thực hiện
1. Mở notebook bằng Google Colab.
2. Tải bộ dữ liệu FGVC-Aircraft.
3. Tiền xử lý dữ liệu và tạo DataLoader.
4. Khởi tạo EfficientNet-B0 pretrained.
5. Huấn luyện mô hình trong 5 Epoch.
6. Đánh giá trên tập Validation và Test.
7. Thực hiện dự đoán ảnh máy bay.

## 7. Kết luận
Mô hình EfficientNet-B0 pretrained đã được áp dụng thành công cho bài toán phân loại 100 loại máy bay thuộc bộ dữ liệu FGVC-Aircraft. Sau 5 Epoch huấn luyện, mô hình đạt độ chính xác 39.81% trên tập Validation và 41.82% trên tập Test.

Kết quả cho thấy phương pháp Transfer Learning có khả năng khai thác các đặc trưng đã học từ ImageNet để giải quyết bài toán phân loại hình ảnh máy bay.
