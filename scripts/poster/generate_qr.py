"""Generate QR code for the LessWrong post."""
import qrcode

url = "https://www.lesswrong.com/posts/pxC2RAeoBrvK8ivMf/models-have-linear-representations-of-what-tasks-they-like-1"
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=20, border=2)
qr.add_data(url)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
out = "docs/poster/assets/qr_lesswrong.png"
img.save(out)
print(f"Saved: {out}")
