import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from neural_style_transfer import run_style_transfer, image_loader, cnn, cnn_normalization_mean, cnn_normalization_std

# === Biến toàn cục ===
content_path = ""
style_path = ""

# === Giao diện chính ===
root = tk.Tk()
root.title("🎨 Neural Style Transfer - AI Art Generator")
root.geometry("1100x750")
root.configure(bg="#fafafa")

# === Placeholder ảnh xám ===
placeholder_image = ImageTk.PhotoImage(Image.new("RGB", (300, 300), color="#CCCCCC"))

# === Tiêu đề ===
title_label = tk.Label(
    root,
    text="🎨 Neural Style Transfer",
    font=("Segoe UI", 30, "bold"),  # font to hơn một chút
    fg="#2c3e50",
    bg="#fafafa",  # cùng màu với nền chính
    pady=20
)
title_label.pack()


# === Frame chứa các nút ===
frame_buttons = tk.Frame(root, bg="#fafafa")
frame_buttons.pack(pady=10)

btn_content = tk.Button(
    frame_buttons, text="📁 Upload Content Image",
    command=lambda: upload_image('content'),
    bg="#3399ff", fg="black",
    activebackground="#CC99FF",  # khi hover
    font=("Segoe UI", 12, "bold"),
    width=24, height=2,
    bd=0, relief="flat", highlightthickness=0
)
btn_content.grid(row=0, column=0, padx=15)

btn_style = tk.Button(
    frame_buttons, text="🎨 Upload Style Image",
    command=lambda: upload_image('style'),
    bg="#2dbe60", fg="black",
    activebackground="#CC99FF",
    font=("Segoe UI", 12, "bold"),
    width=24, height=2,
    bd=0, relief="flat", highlightthickness=0
)
btn_style.grid(row=0, column=1, padx=15)

btn_run = tk.Button(
    root, text="✨ Run Style Transfer",
    command=lambda: do_style_transfer(),
    bg="#e12828", fg="white",
    activebackground="#CD2626",
    font=("Segoe UI", 13, "bold"),
    width=26, height=2,
    bd=0, relief="flat", highlightthickness=0
)
btn_run.pack(pady=20)

# === Frame hiển thị hình ảnh ===
frame_images = tk.Frame(root, bg="#fafafa")
frame_images.pack(pady=10)

# Label hiển thị hình ảnh
label_content = tk.Label(frame_images, text="Content Image\n(300x300)", image=placeholder_image, compound="top", bg="#e0e0e0", width=310, height=320)
label_content.grid(row=0, column=0, padx=20)

label_style = tk.Label(frame_images, text="Style Image\n(300x300)", image=placeholder_image, compound="top", bg="#e0e0e0", width=310, height=320)
label_style.grid(row=0, column=1, padx=20)

label_result = tk.Label(frame_images, text="Result Image\n(300x300)", image=placeholder_image, compound="top", bg="#e0e0e0", width=310, height=320)
label_result.grid(row=0, column=2, padx=20)


# === Hàm upload ảnh nội dung hoặc phong cách ===
def upload_image(image_type):
    global content_path, style_path

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        if image_type == 'content':
            content_path = file_path
            label_content.config(image=img_tk)
            label_content.image = img_tk
        elif image_type == 'style':
            style_path = file_path
            label_style.config(image=img_tk)
            label_style.image = img_tk


# === Hàm xử lý NST ===
def do_style_transfer():
    if not content_path or not style_path:
        messagebox.showerror("Lỗi", "Vui lòng chọn cả ảnh nội dung và ảnh phong cách!")
        return

    try:
        content_image = image_loader(content_path)
        style_image = image_loader(style_path)
        input_img = content_image.clone()

        output = run_style_transfer(
            cnn, cnn_normalization_mean, cnn_normalization_std,
            content_image, style_image, input_img
        )

        result_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        result_img = result_img.resize((300, 300))
        result_tk = ImageTk.PhotoImage(result_img)

        label_result.config(image=result_tk)
        label_result.image = result_tk

    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi:\n{str(e)}")


root.mainloop()

