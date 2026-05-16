Các lỗi cần cập nhật:

# Lỗi logic module AI:
- Sản phẩm nó sẽ ưu tiên hình dáng và màu --> màu --> hình dáng
- Nếu gửi 1 bức ảnh y đúc thì chỉ hiện 99% bởi vì ảnh có thể mờ, khác size dẫn đến nó không phải 100%

# Lỗi UI

# Thiếu chức năng
- Thiếu thêm vào giỏ hàng nhưng mà chưa cần thiết lắm

# câu hỏi bổ sung
- Đã finetune model như thế nào? cách hoạt động của nó ra làm sao?
- Vì sao lại sử dụng ResNet50 thay vì các module khác

# Note
- nếu như lỗi sửa chưa được thì sẽ ghi thêm là website làm về hướng tìm kiếm những sản phẩm cho khách hàng và show ra những mẫu áo khớp vibe mà người dùng có thể mua thay thế.

# test case
ảnh không phải dataset trong sàn tmdt
ảnh là quần áo
ảnh chụp bị mờ

---

Bộ test này chia thành 4 nhóm chính: Độ chính xác AI (Chí mạng), Chức năng
hệ thống, Giao diện (UI/UX) và Trường hợp ngoại lệ (Edge Cases).

# Nhóm 1: Kiểm tra độ chính xác của AI (AI Accuracy Testing)

Đây là phần quan trọng nhất để giải quyết vấn đề "Áo đen ra váy trắng" của bạn.

| STT     | Tên Test Case                 | Mô tả các bước                                                     | Kết quả mong đợi                                                                               |
| :------ | :---------------------------- | :----------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **1.1** | **Khớp sản phẩm gốc**         | Upload một ảnh có sẵn trong thư mục `datasets`.                    | Sản phẩm đó phải hiện ở vị trí số 1 với độ khớp ≈ 99-100%.                                     |
| **1.2** | **Ưu tiên Màu sắc**           | Upload ảnh áo thun màu Đỏ. Trong DB có áo thun Đỏ và áo thun Xanh. | Các sản phẩm màu Đỏ phải xếp trên màu Xanh. Độ khớp \> 80%.                                    |
| **1.3** | **Ưu tiên Kiểu dáng**         | Upload ảnh một chiếc Quần Jean.                                    | Kết quả trả về phải là Quần, không được lẫn lộn sang Áo hay Váy.                               |
| **1.4** | **Phân biệt Trơn & Họa tiết** | Upload ảnh Áo thun đen trơn.                                       | Kết quả đầu tiên phải là áo trơn. Các áo có hoa văn/chữ phải bị đẩy xuống dưới hoặc có % thấp. |
| **1.5** | **Tìm kiếm đa đặc trưng**     | Upload ảnh Váy hoa màu xanh lá.                                    | Kết quả phải ưu tiên: Váy + Màu xanh + Có hoa văn.                                             |

# Nhóm 2: Kiểm tra Chức năng (Functional Testing)

Kiểm tra các nút bấm, luồng dữ liệu và phân trang.

| STT     | Tên Test Case               | Mô tả các bước                                                  | Kết quả mong đợi                                                             |
| :------ | :-------------------------- | :-------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **2.1** | **Upload đa phương thức**   | Test upload bằng 3 cách: Chọn file, Kéo thả, và Paste (Ctrl+V). | Cả 3 cách đều phải hiển thị được ảnh Preview và sẵn sàng Analyze.            |
| **2.2** | **Phân trang (Pagination)** | Nhấn vào trang 2, trang 3 tại Trang chủ.                        | Danh sách sản phẩm thay đổi, chỉ hiển thị đúng 50 sản phẩm mới.              |
| **2.3** | **Lọc theo Category**       | Chọn danh mục "Shirt" ở sidebar.                                | Chỉ các sản phẩm thuộc nhóm Shirt hiện ra, các nhóm khác biến mất.           |
| **2.4** | **Sắp xếp (Sorting)**       | Chọn sắp xếp theo giá "Price Low to High".                      | Sản phẩm phải được sắp xếp đúng theo giá tăng dần trong CSV.                 |
| **2.5** | **Truy vấn Database**       | Bấm Analyze một ảnh.                                            | Kết quả phải hiển thị đầy đủ: Tên, Giá, Category lấy từ file `products.csv`. |

# Nhóm 3: Kiểm tra Giao diện & Responsive (UI/UX Testing)

Đảm bảo Layout 5 cột và 3 cột không bị lỗi.

| STT     | Tên Test Case             | Mô tả các bước                                               | Kết quả mong đợi                                                     |
| :------ | :------------------------ | :----------------------------------------------------------- | :------------------------------------------------------------------- |
| **3.1** | **Desktop Layout**        | Mở trang chủ trên trình duyệt máy tính (màn hình \> 1200px). | Grid sản phẩm phải hiện đúng 5 cột/hàng.                             |
| **3.2** | **Search Results Layout** | Thực hiện tìm kiếm bằng AI.                                  | Kết quả tìm kiếm AI phải hiện đúng 3 cột/hàng (không bị nhảy lên 5). |
| **3.3** | **Tablet Responsive**     | Thu nhỏ trình duyệt về chiều rộng ≈ 768px.                   | Grid sản phẩm tự động chuyển về 3 cột/hàng.                          |
| **3.4** | **Mobile Responsive**     | Thu nhỏ trình duyệt về chiều rộng ≈ 480px.                   | Grid sản phẩm tự động chuyển về 2 cột/hàng.                          |
| **3.5** | **Tỉ lệ hình ảnh**        | Quan sát các ảnh có kích thước khác nhau (dọc/ngang).        | Ảnh không bị bóp méo, giữ đúng tỉ lệ khung hình (aspect-ratio 3/4).  |

# Nhóm 4: Kiểm tra trường hợp ngoại lệ (Edge Cases)

Kiểm tra độ bền của hệ thống khi gặp dữ liệu lạ.

| STT     | Tên Test Case                 | Mô tả các bước                                                 | Kết quả mong đợi                                                                 |
| :------ | :---------------------------- | :------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **4.1** | **Ảnh không phải thời trang** | Upload ảnh một con mèo hoặc một cái ô tô.                      | AI vẫn chạy nhưng trả về độ khớp (Match %) rất thấp (\< 20-30%).                 |
| **4.2** | **File không hợp lệ**         | Cố tình upload một file `.txt` hoặc `.pdf` (nếu có thể).       | Hệ thống chặn file và thông báo: "Định dạng file không hỗ trợ".                  |
| **4.3** | **Ảnh dung lượng lớn**        | Upload ảnh độ phân giải 4K (nặng \> 10MB).                     | Server không bị crash, vẫn xử lý được (có thể chậm hơn chút).                    |
| **4.4** | **Ảnh môi trường phức tạp**   | Upload ảnh người mẫu mặc quần áo đứng giữa đường phố đông đúc. | AI phải lấy được màu của quần áo ở giữa ảnh, không lấy màu của xe cộ xung quanh. |
| **4.5** | **Database trống**            | Xóa sạch nội dung file `products.csv`.                         | Trang web không bị lỗi trắng trang, hiện thông báo "No products found".          |

## Lời khuyên khi thực hiện Test:

1.  Dùng công cụ Developer Tools (F12): Mở tab Console để xem có lỗi JavaScript
    nào không và tab Network để xem API /search mất bao lâu để phản hồi.
2.  Chuẩn bị dữ liệu Test: Bạn nên có một bộ sưu tập ảnh "khó" (ví dụ: áo đen
    nhưng chụp trong tối, hoặc váy trắng nhưng nền cũng màu trắng) để thử thách
    mô hình.
3.  Ghi lại Accuracy Score: Hãy lập một bảng Excel ghi lại: Ảnh gốc - Kết quả AI
    tìm thấy - % khớp. Nếu ảnh đúng nằm trong Top 3, mô hình của bạn đạt yêu
    cầu.

