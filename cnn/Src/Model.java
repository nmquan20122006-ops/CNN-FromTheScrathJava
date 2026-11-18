package Src;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/*
 * Lưu / tải toàn bộ mạng (kiến trúc + trọng số đã học) ra một file.
 */
public class Model {
    public static void save(Network net, String path) {
        net.clearCache(); // không lưu input/mask của batch cuối -> file nhỏ hơn nhiều
        try (ObjectOutputStream out = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeObject(net);
            System.out.println("Da luu model vao " + path);
        } catch (IOException e) {
            throw new RuntimeException("Luu model that bai: " + e.getMessage(), e);
        }
    }

    public static Network load(String path) {
        try (ObjectInputStream in = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(path)))) {
            return (Network) in.readObject();
        } catch (Exception e) {
            throw new RuntimeException("Doc model that bai: " + e.getMessage(), e);
        }
    }
}
