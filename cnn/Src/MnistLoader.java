package Src;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

/*
 * Đọc bộ dữ liệu MNIST ở định dạng IDX (có thể là .gz hoặc file thường).
 *
 * Images magic number = 2051: count, rows, cols, rồi count*rows*cols byte pixel.
 * Labels magic number = 2049: count, rồi count byte nhãn.
 *
 * Pixel được chia 255 để đưa về [0, 1].
 */
public class MnistLoader {
    public static class Data {
        public final Tensor x; // (n, 1, 28, 28)
        public final int[] y;  // nhãn 0..9

        public Data(Tensor x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }

    public static Data load(String imagePath, String labelPath, int limit) {
        try (DataInputStream img = open(imagePath);
             DataInputStream lbl = open(labelPath)) {

            int magicImages = img.readInt();
            if (magicImages != 2051) {
                throw new IOException("Khong phai file IDX images: " + imagePath);
            }
            int count = img.readInt();
            int rows = img.readInt();
            int cols = img.readInt();

            int magicLabels = lbl.readInt();
            if (magicLabels != 2049) {
                throw new IOException("Khong phai file IDX labels: " + labelPath);
            }
            int labelCount = lbl.readInt();

            if (count != labelCount) {
                throw new IOException("So anh va so nhan khong khop: " + count + " != " + labelCount);
            }

            int n = (limit > 0) ? Math.min(limit, count) : count;
            Tensor x = new Tensor(n, 1, rows, cols);
            int[] y = new int[n];

            byte[] pixels = new byte[rows * cols];
            for (int i = 0; i < n; i++) {
                img.readFully(pixels);
                int offset = i * pixels.length;
                for (int p = 0; p < pixels.length; p++) {
                    x.data[offset + p] = (pixels[p] & 0xFF) / 255.0;
                }
                y[i] = lbl.readUnsignedByte();
            }
            return new Data(x, y);

        } catch (IOException e) {
            throw new RuntimeException("Doc MNIST that bai: " + e.getMessage(), e);
        }
    }

    private static DataInputStream open(String path) throws IOException {
        InputStream in = new FileInputStream(path);
        if (path.endsWith(".gz")) {
            in = new GZIPInputStream(in);
        }
        return new DataInputStream(new BufferedInputStream(in));
    }
}
