package MSP;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class Train_MSP {
	public static void main(String[] args) throws Exception {
		long start = System.currentTimeMillis();
		Train train = new Train();
		System.out.println("k=" + train.m);
		System.out.println("margin=" + train.margin);
		System.out.println("rate=" + train.rate);
		train.run();

		long end = System.currentTimeMillis();
		System.out.println("time:" + (end - start) + "ms");
	}
}

class Train {
	int rel_num = 0, ent_num = 0;
	int m = Utils.DIMENSION;
	double rate = 0.0001;
	double margin = 2;
	boolean L1_flag = true;
	boolean method = false;
	HashMap<Integer, SimpleMatrix> entity2vec = new HashMap<Integer, SimpleMatrix>();
	HashMap<Integer, SimpleMatrix> relation2vec = new HashMap<Integer, SimpleMatrix>();
	Map<String, Integer> rel_id = new HashMap<String, Integer>();
	Map<String, Integer> ent_id = new HashMap<String, Integer>();

	HashMap<Integer, SimpleMatrix> matrix_wr = new HashMap<>();
	HashMap<Integer, SimpleMatrix> matrix_r = new HashMap<Integer, SimpleMatrix>();
	HashMap<Integer, SimpleMatrix> matrix_e = new HashMap<Integer, SimpleMatrix>();
	SimpleMatrix matrix_I = new SimpleMatrix(m, m);
	Map<Integer, Map<Integer, Map<Integer, Integer>>> tuple;
	List<Integer> ent_h = new ArrayList<Integer>();
	List<Integer> ent_t = new ArrayList<Integer>();
	List<Integer> rel = new ArrayList<Integer>();
	HashMap<Integer, SimpleMatrix> entity_tmp, relation_tmp;
	HashMap<Integer, SimpleMatrix> matrix_wr_tmp, matrix_r_tmp, matrix_e_tmp;
	Map<Integer, Map<Integer, ArrayList<Integer>>> left_entity, right_entity;
	Map<Integer, ArrayList<Integer>> rel_tail, rel_head;
	Map<Integer, Double> left_mean, right_mean;

	double loss = 0;
	double min = Double.MAX_VALUE;
	int min_epoch;

	void run() throws IOException {
		prepare();
		System.out.println("train prepared");

		int nbatches = 100;
		int nepoch = Utils.NEPOCH;// 1000
		int batchsize = ent_h.size() / nbatches;
		Random rand = new Random();
		entity_tmp = Utils.clone(entity2vec);
		relation_tmp = Utils.clone(relation2vec);
		matrix_e_tmp = Utils.clone(matrix_e);
		matrix_r_tmp = Utils.clone(matrix_r);
		matrix_wr_tmp = Utils.clone(matrix_wr);

		for (int epoch = 0; epoch < nepoch; epoch++) {
			loss = 0;
			for (int batch = 0; batch < nbatches; batch++) {
				for (int k = 0; k < batchsize; k++) {
					int i = rand.nextInt(ent_h.size());
					int j = rand.nextInt(ent_num);
					double pr = right_mean.get(rel.get(i)) / (right_mean.get(rel.get(i)) + left_mean.get(rel.get(i)));
					if (method) {
						pr = 0.5;
					}
					if (Math.random() < pr) {

						while (tuple.get(ent_h.get(i)).get(rel.get(i)).get(j) != null) {
							j = rand.nextInt(ent_num);
						}
						calcLoss(ent_h.get(i), ent_t.get(i), rel.get(i), ent_h.get(i), j, rel.get(i));
					} else {

						while (tuple.get(j) != null && tuple.get(j).get(rel.get(i)) != null
								&& tuple.get(j).get(rel.get(i)).get(ent_t.get(i)) != null) {
							j = rand.nextInt(ent_num);
						}
						calcLoss(ent_h.get(i), ent_t.get(i), rel.get(i), j, ent_t.get(i), rel.get(i));
					}
					relation_tmp.put(rel.get(i), Utils.Nomalize(relation_tmp.get(rel.get(i))));
					entity_tmp.put(ent_h.get(i), Utils.Nomalize(entity_tmp.get(ent_h.get(i))));
					entity_tmp.put(ent_t.get(i), Utils.Nomalize(entity_tmp.get(ent_t.get(i))));
					entity_tmp.put(j, Utils.Nomalize(entity_tmp.get(j)));
					matrix_e_tmp.put(ent_h.get(i), Utils.Nomalize(matrix_e_tmp.get(ent_h.get(i))));
					matrix_e_tmp.put(ent_t.get(i), Utils.Nomalize(matrix_e_tmp.get(ent_t.get(i))));
					matrix_e_tmp.put(j, Utils.Nomalize(matrix_e_tmp.get(j)));
					matrix_r_tmp.put(rel.get(i), Utils.Nomalize(matrix_r_tmp.get(rel.get(i))));
					matrix_wr_tmp.put(rel.get(i), Utils.Nomalize(matrix_wr_tmp.get(rel.get(i))));
				}
				entity2vec = Utils.clone(entity_tmp);
				relation2vec = Utils.clone(relation_tmp);
				matrix_e = Utils.clone(matrix_e_tmp);
				matrix_r = Utils.clone(matrix_r_tmp);
				matrix_wr = Utils.clone(matrix_wr_tmp);
			}

			String cout = "epoch:" + epoch + " " + loss + "\r\n";
			System.out.println(cout);

			String path = "MSP";
			new File(path).mkdirs();
			FileWriter writer1 = new FileWriter(path + "/entity2vec.bern", false);
			FileWriter writer2 = new FileWriter(path + "/relation2vec.bern", false);
			FileWriter writer3 = new FileWriter(path + "/matrix_wr.bern", false);
			FileWriter writer4 = new FileWriter(path + "/matrix_e.bern", false);
			FileWriter writer5 = new FileWriter(path + "/matrix_r.bern", false);

			for (int i = 0; i < ent_num; i++) {
				print(writer1, entity2vec.get(i));
				print(writer4, matrix_e.get(i));
			}

			for (int i = 0; i < rel_num; i++) {
				print(writer2, relation2vec.get(i));
				print(writer3, matrix_wr.get(i));
				print(writer5, matrix_r.get(i));
			}

			writer1.close();
			writer2.close();
			writer3.close();
			writer4.close();
			writer5.close();
			if (loss < min) {
				min = loss;
				min_epoch = epoch;
			}
		}
		System.out.println("min_epoch:" + min_epoch + "  loss:" + min);
	}

	double calcScore(int eh, int et, int rel) {
		SimpleMatrix matrix_rh = matrix_r.get(rel).mult(matrix_e.get(eh)).plus(matrix_I);
		SimpleMatrix matrix_rt = matrix_r.get(rel).mult(matrix_e.get(et)).plus(matrix_I);
		SimpleMatrix h_c = entity2vec.get(eh)
				.minus(matrix_wr.get(rel).scale(matrix_wr.get(rel).transpose().mult(entity2vec.get(eh)).get(0)));
		SimpleMatrix t_c = entity2vec.get(et)
				.minus(matrix_wr.get(rel).scale(matrix_wr.get(rel).transpose().mult(entity2vec.get(et)).get(0)));
		SimpleMatrix h_p = matrix_rh.mult(h_c);
		SimpleMatrix t_p = matrix_rt.mult(t_c);
		SimpleMatrix vec_tmp = h_p.plus(relation2vec.get(rel)).minus(t_p);

		double score = 0.0;
		if (L1_flag)
			for (int i = 0; i < m; i++) {
				score += Math.abs(vec_tmp.get(i, 0));
			}
		else
			for (int i = 0; i < m; i++) {
				score += vec_tmp.get(i, 0) * vec_tmp.get(i, 0);
			}
		return score;
	}

	void gradient(int eh, int et, int rel, int belta) {

		SimpleMatrix matrix_rh = matrix_r.get(rel).mult(matrix_e.get(eh)).plus(matrix_I);
		SimpleMatrix matrix_rt = matrix_r.get(rel).mult(matrix_e.get(et)).plus(matrix_I);
		SimpleMatrix h_c = entity2vec.get(eh)
				.minus(matrix_wr.get(rel).scale(matrix_wr.get(rel).transpose().mult(entity2vec.get(eh)).get(0)));
		SimpleMatrix t_c = entity2vec.get(et)
				.minus(matrix_wr.get(rel).scale(matrix_wr.get(rel).transpose().mult(entity2vec.get(et)).get(0)));
		SimpleMatrix h_p = matrix_rh.mult(h_c);
		SimpleMatrix t_p = matrix_rt.mult(t_c);
		SimpleMatrix vec_tmp = h_p.plus(relation2vec.get(rel)).minus(t_p);

		double tmp1 = matrix_wr.get(rel).transpose().mult(entity2vec.get(eh)).get(0);
		double tmp2 = matrix_wr.get(rel).transpose().mult(entity2vec.get(et)).get(0);
		for (int i = 0; i < m; i++) {
			double tmp_Dh1 = 0;
			double tmp_Dh2 = 0;
			double xx = 2 * vec_tmp.get(i);
			if (L1_flag)
				if (xx > 0)
					xx = 1;
				else
					xx = -1;
			relation_tmp.get(rel).set(i, relation_tmp.get(rel).get(i) - belta * rate * xx);

			for (int j = 0; j < m; j++) {
				double x = 2 * vec_tmp.get(j);
				if (L1_flag)
					if (x > 0)
						x = 1;
					else
						x = -1;
				tmp_Dh1 += x * matrix_rh.get(j, i);
				tmp_Dh2 += x * matrix_rt.get(j, i);
				matrix_r_tmp.get(rel).set(i, matrix_r_tmp.get(rel).get(i) - belta * rate * xx
						* (h_c.get(j) * matrix_e.get(eh).get(j) - t_c.get(j) * matrix_e.get(et).get(j)));
				matrix_e_tmp.get(eh).set(i,
						matrix_e_tmp.get(eh).get(i) - belta * rate * x * h_c.get(i) * matrix_r.get(rel).get(j));
				matrix_e_tmp.get(et).set(i,
						matrix_e_tmp.get(et).get(i) + belta * rate * x * t_c.get(i) * matrix_r.get(rel).get(j));
			}
			entity_tmp.get(eh).set(i, entity_tmp.get(eh).get(i) - belta * rate * tmp_Dh1);
			entity_tmp.get(et).set(i, entity_tmp.get(et).get(i) + belta * rate * tmp_Dh2);
			matrix_wr_tmp.get(rel).set(i,
					matrix_wr_tmp.get(rel).get(i) + belta * rate * (tmp_Dh1 * tmp1 - tmp_Dh2 * tmp2));
			for (int j = 0; j < m; j++) {
				entity_tmp.get(eh).set(j, entity_tmp.get(eh).get(j)
						+ belta * rate * tmp_Dh1 * matrix_wr.get(rel).get(j) * matrix_wr.get(rel).get(i));
				entity_tmp.get(et).set(j, entity_tmp.get(et).get(j)
						- belta * rate * tmp_Dh2 * matrix_wr.get(rel).get(j) * matrix_wr.get(rel).get(i));
				matrix_wr_tmp.get(rel).set(j, matrix_wr_tmp.get(rel).get(j) + belta * rate * matrix_wr.get(rel).get(i)
						* (tmp_Dh1 * entity2vec.get(eh).get(j) - tmp_Dh2 * entity2vec.get(et).get(j)));
			}
		}
	}

	void calcLoss(int eh_1, int et_1, int rel_1, int eh_2, int et_2, int rel_2) {
		double score1 = calcScore(eh_1, et_1, rel_1);
		double score2 = calcScore(eh_2, et_2, rel_2);
		if (margin + score1 > score2) {
			loss += margin + score1 - score2;
			gradient(eh_1, et_1, rel_1, 1);
			gradient(eh_2, et_2, rel_2, -1);
		}
	}

	void print(FileWriter f, SimpleMatrix m) throws IOException {
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++)
				f.write(m.get(i, j) + "\t");
		}
		f.write("\r\n");
		f.flush();
	}

	void prepare() throws NumberFormatException, IOException {
		String path = "data/FB15k/";
		BufferedReader reader = new BufferedReader(new FileReader(path + "entity2id.txt"));
		String line;
		double[] vector;
		int count = 0;

		while ((line = reader.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			ent_id.put(tokens[0], Integer.parseInt(tokens[1]));
			ent_num++;
		}
		reader.close();

		reader = new BufferedReader(new FileReader(path + "relation2id.txt"));
		while ((line = reader.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			rel_id.put(tokens[0], Integer.parseInt(tokens[1]));
			rel_num++;
		}
		reader.close();

		 reader = new BufferedReader(new FileReader("transE/FB15k/" + Utils.DIMENSION +
		 "/entity2vec.bern"));
		 while ((line = reader.readLine()) != null) {
		 String[] tokens = line.split("\\s+");
		 vector = new double[m];
		 for (int i = 0; i < m; i++) {
		 vector[i] = Double.parseDouble(tokens[i]);
		 }
		 entity2vec.put(count, new SimpleMatrix(m, 1, true, vector));
		 count++;
		 }
		 reader.close();
		
		 reader = new BufferedReader(new FileReader("transE/FB15k/" + Utils.DIMENSION +
		 "/relation2vec.bern"));
		 count = 0;
		 while ((line = reader.readLine()) != null) {
		 String[] tokens = line.split("\\s+");
		 vector = new double[m];
		 for (int i = 0; i < m; i++) {
		 vector[i] = Double.parseDouble(tokens[i]);
		 }
		 relation2vec.put(count, new SimpleMatrix(m, 1, true, vector));
		 count++;
		 }
		 reader.close();
		
		 for (int i = 0; i < rel_num; i++) {
		 matrix_r.put(i, new SimpleMatrix(m, 1));
		 matrix_wr.put(i, new SimpleMatrix(m, 1));
		 for (int ii = 0; ii < m; ii++) {
		 matrix_r.get(i).set(ii, 2 * Math.random() - 1);
		 matrix_wr.get(i).set(ii, 2 * Math.random() - 1);
		 }
		 }
		
		 for (int i = 0; i < ent_num; i++) {
		 matrix_e.put(i, new SimpleMatrix(1, m));
		 for (int ii = 0; ii < m; ii++) {
		 matrix_e.get(i).set(ii, 2 * Math.random() - 1);
		 }
		 }

		for (int ii = 0; ii < m; ii++) {
			for (int jj = 0; jj < m; jj++) {
				if (ii == jj) {
					matrix_I.set(ii, jj, 1);
				}
			}
		}

		reader = new BufferedReader(new FileReader(path + "train.txt"));
		String relName, headName, tailName;
		tuple = new HashMap<>();
		left_entity = new HashMap<>();
		right_entity = new HashMap<>();
		rel_tail = new HashMap<>();
		rel_head = new HashMap<>();
		while ((line = reader.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			relName = tokens[Utils.RELATION_IDX];
			headName = tokens[Utils.Head_IDX];
			tailName = tokens[Utils.TAIL_IDX];
			if (ent_id.get(headName) == null) {
				System.out.println("miss entity:" + headName);
			}
			if (ent_id.get(tailName) == null) {
				System.out.println("miss entity:" + tailName);
			}
			if (rel_id.get(relName) == null) {
				rel_id.put(relName, rel_num);
				rel_num++;
			}
			int x, y, z;
			x = ent_id.get(headName);
			y = ent_id.get(tailName);
			z = rel_id.get(relName);
			ent_h.add(x);
			ent_t.add(y);
			rel.add(z);
			if (tuple.get(x) == null) {
				tuple.put(x, new HashMap<>());
			}
			if (tuple.get(x).get(z) == null) {
				tuple.get(x).put(z, new HashMap<>());
			}
			tuple.get(x).get(z).put(y, 1);

			if (left_entity.get(z) == null) {
				left_entity.put(z, new HashMap<>());
			}
			if (left_entity.get(z).get(x) == null) {
				left_entity.get(z).put(x, new ArrayList<>());
			}
			left_entity.get(z).get(x).add(y);

			if (right_entity.get(z) == null) {
				right_entity.put(z, new HashMap<>());
			}
			if (right_entity.get(z).get(y) == null) {
				right_entity.get(z).put(y, new ArrayList<>());
			}
			right_entity.get(z).get(y).add(x);

			if (rel_tail.get(z) == null) {
				ArrayList<Integer> ent = new ArrayList<>();
				ent.add(y);
				rel_tail.put(z, ent);
			} else {
				ArrayList<Integer> ent = rel_tail.get(z);
				ent.add(y);
				rel_tail.put(z, ent);
			}
			if (rel_head.get(z) == null) {
				ArrayList<Integer> ent = new ArrayList<>();
				ent.add(x);
				rel_head.put(z, ent);
			} else {
				ArrayList<Integer> ent = rel_head.get(z);
				ent.add(x);
				rel_head.put(z, ent);
			}

		}
		reader.close();

		left_mean = new HashMap<>();
		right_mean = new HashMap<>();
		Map<Integer, ArrayList<Integer>> tmp;
		for (int i = 0; i < rel_num; i++) {
			double sum = 0;
			tmp = left_entity.get(i);
			for (Integer ii : tmp.keySet()) {
				sum += tmp.get(ii).size();
			}
			left_mean.put(i, sum / tmp.size());

			sum = 0;
			tmp = right_entity.get(i);
			for (Integer ii : tmp.keySet()) {
				sum += tmp.get(ii).size();
			}
			right_mean.put(i, sum / tmp.size());
		}
	}
}
