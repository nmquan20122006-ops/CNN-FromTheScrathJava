package Layer;
import Src.Tensor;
import java.io.Serializable;
public interface Layer extends Serializable {
    Tensor forward(Tensor input);
    Tensor backward(Tensor dOut); // dOut là đạo hàm được truyền từ layer phía sau
    void update(double learningRate);

    // Chọn cách cập nhật trọng số: "sgd" (mặc định) hoặc "adam"
    default void setOptimizer(String name) {}

    // Xoá trạng thái tạm của lần forward gần nhất trước khi lưu model
    default void clearCache() {}
}
