package demo01;

public class AConF implements AFruitAndVegetables{

    @Override
    public Fruit CreateF() {
        return new Apple();
    }

    @Override
    public Vegetables CreateV() {
        return new Cabbage();
    }
}
