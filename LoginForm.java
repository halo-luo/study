package demo01;

public class LoginForm {
	
	UserDAO dao = new UserDAO();
	
	void init(){
		System.out.println("初始化成功！");
	}
	
	void display(){
		System.out.println("display方法！");
	}
	
	void validate(){
		
		dao.findUser("username", "userPassword");
		System.out.println("validate方法！");
		
	}
	
}
