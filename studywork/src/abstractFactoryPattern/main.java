package abstractFactoryPattern;

public class main {
    public static void main(String args[]){

        AConF Acon = new AConF();
        Fruit apple = Acon.CreateF();

        Vegetables cabbage = Acon.CreateV();
        System.out.println("这是A工厂的产物");
        apple.eat();
        cabbage.eat();

        BConF Bcon = new BConF();
        Fruit banana = Bcon.CreateF();
        Vegetables tomato = Bcon.CreateV();
        System.out.println("这是B工厂的产物");
        banana.eat();
        tomato.eat();

    }
}
