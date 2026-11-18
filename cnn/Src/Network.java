package Src;
import Layer.Layer;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
public class Network implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<Layer> _layers = new ArrayList<>();
    public void addLayer(Layer layer){
        _layers.add(layer);
    }
    public Tensor forward(Tensor input){
        Tensor x = input;
        for(Layer layer:_layers){
            x = layer.forward(x);
        }
        return x;
    }
    public Tensor backward(Tensor gradOutput){
        Tensor g = gradOutput;
        for(int i = _layers.size() -1; i>=0;i--){
            g = _layers.get(i).backward(g);
        }
        return g;
    }
    public void update(double learningRate){
        for(Layer layer: _layers){
            layer.update(learningRate);
        }
    }

    // Chọn optimizer cho toàn bộ layer có tham số (Conv2D, Dense)
    public void setOptimizer(String name){
        for(Layer layer: _layers){
            layer.setOptimizer(name);
        }
    }

    // Dọn cache trước khi serialize; sau khi tải model phải forward lại mới backward được
    public void clearCache(){
        for(Layer layer: _layers){
            layer.clearCache();
        }
    }
}
