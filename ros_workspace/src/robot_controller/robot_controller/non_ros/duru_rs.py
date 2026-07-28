import pyrealsense2 as rs  # cam check, getthe views and calculate 3D
import numpy as np  # matrixleri hızlı işleme kütüpü
import cv2

# Fare tıklama koordinatlarını tutacak değişkenler
clicked_u, clicked_v = -1, -1


def mouse_callback(event, x, y, flags, param):#opencv nin fare haraketlerini dinlemek için kullandığı standart funct yapısı
  global clicked_u, clicked_v #normal global yapısı 
  if event == cv2.EVENT_LBUTTONDOWN:#eğer sol tuş basarsa
    clicked_u, clicked_v = x, y


pipeline = rs.pipeline()  # bilgsy ve cam arasında bir veri boru hattı oluşturur
config = rs.config()  # boş ayar dosyası ki came e ayar yapılabilsin

config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)  # cam e derinlik(Z) sensörünü açmasını söyler ve fps , format ve çözünürlüğü ayarlar
config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)  # renkli(RBG) açmasını söyler

# Kamerayı çalıştır ve yukarıdaki tüm ayarları donanaıma gönderir
profile = pipeline.start(config)

# 2. Hizalama (Alignment) Nesnesini Oluştur
# Derinlik sensörü ile RGB lensi fiziksel olarak farklı yerlerdedir (aralarında birkaç cm vardır).
# Bu işlem, derinlik haritasını renkli kameranın bakış açısına kaydırarak eşleştirir.
align_to = rs.stream.color
align = rs.align(
    align_to
)  # hizalama işlemini karaler geldikçe yapacak aracı oluşturur

# OpenCV penceresini aç ve fare fonksiyonunu bağla
cv2.namedWindow("RealSense Canli Yayin")
cv2.setMouseCallback("RealSense Canli Yayin", mouse_callback)

try:
  # Kameranın ışığa alışması için ilk birkaç kareyi atla
  for i in range(10):
    pipeline.wait_for_frames()

  while True:
    # Kareleri al ve hizalama işlemini uygula
    frames = (
        pipeline.wait_for_frames()
    )  # renk ve derinliği içeren en taze kareyi alır
    aligned_frames = align.process(
        frames
    )  # alınan taze framei yukaradı kurulan hizalama aracına sokar

    aligned_depth_frame = (
        aligned_frames.get_depth_frame()
    )  # sadece derinlik bilgisini taşşıyan kareyi alır
    color_frame = aligned_frames.get_color_frame()  # renk.

    if not aligned_depth_frame or not color_frame:
      print("Kare okunamadı.")
      continue

    # 3. Kameranın İç Parametrelerini (Intrinsics) Çek
    depth_intrin = (
        aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    )
    # kameranın odak uzaklığı, mercek bükülmesi, ve optik merkz gibi fiziksel verileri .eker ve formıldeki K matrixsini oluşturmuş olur.

    # Renkli görüntüyü OpenCV matrisine çevir
    color_image = np.asanyarray(color_frame.get_data())

    # Ekranda bir yere tıklandıysa dinamik olarak o pikseli al
    if clicked_u != -1 and clicked_v != -1:
      u, v = clicked_u, clicked_v

      # 4. Seçilen pikselin derinliğini (Z) metre cinsinden oku
      depth_value = aligned_depth_frame.get_distance(u, v)

      # 5. DEPROJECTION: 2D pikseli ve Z değerini verip 3D konumu al
      depth_point = rs.rs2_deproject_pixel_to_point(
          depth_intrin, [u, v], depth_value
      )

      # Tıklanan yere görsel efekt ekle
      cv2.circle(color_image, (u, v), 5, (0, 255, 0), -1)
      text = f"X:{depth_point[0]:.2f} Y:{depth_point[1]:.2f} Z:{depth_point[2]:.2f}m"
      cv2.putText(
          color_image,
          text,
          (u + 10, v - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

      # Sonuçları yazdır
      print("sonuç")
      print(f"Hedef Piksel (u, v): ({u}, {v})")
      print(f"Kameraya Olan Uzaklık (Z): {depth_value:.3f} metre")
      print(
          "3D Uzay Koordinatı (X, Y, Z):"
          f" ({depth_point[0]:.3f}, {depth_point[1]:.3f}, {depth_point[2]:.3f})"
      )

    # Canlı akışı göster
    cv2.imshow("RealSense Canli Yayin", color_image)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

finally:
  # İşlem bitince kamerayı güvenli bir şekilde kapat
  pipeline.stop()
  cv2.destroyAllWindows()
