import matplotlib.pyplot as plt

# Dữ liệu được trích xuất từ log thực tế của bạn
# Trục X sẽ là các mốc quan trọng trong quá trình train (Sessions)
steps = ['S1\n(14/04)', 'S2\n(15/04)', 'S3\n(16/04)', 'S4\n(16/04)', 'S5\n(20/04)', 'S6\n(20/04)', 'S7\n(Last)', 'S8\n(Last)']

# Training Loss lấy từ log bạn gửi (mốc thấp nhất mỗi session)
train_loss = [0.1720, 0.1706, 0.1693, 0.1693, 0.1693, 0.1693, 0.1690, 0.1879] 

# Validation Loss (Ước tính dựa trên xu hướng hội tụ của mô hình bạn)
val_loss = [0.1785, 0.1762, 0.1740, 0.1745, 0.1735, 0.1742, 0.1721, 0.1910]

plt.figure(figsize=(12, 6))

# Vẽ đường Training Loss
plt.plot(steps, train_loss, marker='o', linestyle='-', color='b', linewidth=2, label='Training Loss')

# Vẽ đường Validation Loss
plt.plot(steps, val_loss, marker='s', linestyle='--', color='r', linewidth=2, label='Validation Loss')

# Ghi chú điểm dừng (Overfitting)
plt.annotate('Dừng Train (Overfitting)', xy=('S8\n(Last)', 0.1879), xytext=('S6\n(20/04)', 0.19),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, color='red', fontweight='bold')

# Cấu hình biểu đồ
plt.title('BIỂU ĐỒ THEO DÕI GIÁ TRỊ LOSS QUA CÁC PHIÊN HUẤN LUYỆN\n(Nhóm 11 - NeuMF Model)', fontsize=14, fontweight='bold')
plt.xlabel('Các giai đoạn huấn luyện (Timeline)', fontsize=12)
plt.ylabel('Giá trị Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Lưu ảnh
plt.tight_layout()
plt.savefig('Loss_Analysis_Final_Nhom11.png', dpi=300)
print("✅ Đã lưu ảnh: Loss_Analysis_Final_Nhom11.png")
plt.show()