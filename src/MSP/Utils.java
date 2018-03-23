package MSP;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

public class Utils {
	public static final int NEPOCH = 1000;
	public static final int DIMENSION = 300;
	public static final int Head_IDX = 0;
	public static final int TAIL_IDX = 1;
	public static final int RELATION_IDX = 2;
	
	public static void getRandomSet(int min, int max, int n, Set<Integer> set) {
		if (n > (max - min) || max < min || set == null) {
			return;
		}
		for (int i = 0; i < n - set.size(); i++) {
			int num = (int) (Math.random() * (max - min)) + min;
			set.add(num);
		}
		if (set.size() < n) {
			getRandomSet(min, max, n, set);
		}
	}
	
	public static SimpleMatrix Nomalize(SimpleMatrix vec){
		double sum = 0.0;
		for (int i = 0; i < DIMENSION; i++) {
			sum += vec.get(i) * vec.get(i);
		}
		sum = Math.sqrt(sum);
		if(sum > 1){
			return vec.divide(sum);
		}
		return vec;
	}
	
	@SuppressWarnings("unchecked")
	public static <T extends Serializable> T clone(T obj){
		
		T clonedObj = null;
		try {
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(baos);
			oos.writeObject(obj);
			oos.close();
			
			ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
			ObjectInputStream ois = new ObjectInputStream(bais);
			clonedObj = (T) ois.readObject();
			ois.close();
			
		}catch (Exception e){
			e.printStackTrace();
		}
		
		return clonedObj;
	}

}