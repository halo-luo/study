package abstractFactoryPattern;

public class BConF implements BFruitAndVegetables{
    @Override
    public Fruit CreateF() {
        return new Banana();
    }

    @Override
    public Vegetables CreateV() {
        return new tomato();
    }
}
