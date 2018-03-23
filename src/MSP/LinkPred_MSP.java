package MSP;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.ejml.simple.SimpleMatrix;

import com.sun.xml.internal.ws.developer.StreamingAttachment;

public class LinkPred_MSP {

	public static int HEAD_IDX = Utils.Head_IDX;
	public static int TAIL_IDX = Utils.TAIL_IDX;
	public static int RELATION_IDX = Utils.RELATION_IDX;
	public static int m =  Utils.DIMENSION;
	public static int relation_num = 0;
	public static int entity_num = 0;
	public static int RIGHT_NUM = 0;
	public static int RANK_NUM = 0;
	public static int F_RIGHT_NUM = 0;
	public static int F_RANK_NUM = 0;
	public static boolean L1_flag = true;
	public static Map<String, Integer> entity2id = new HashMap<String, Integer>();
	public static Map<String, Integer> relation2id = new HashMap<>();
	public static Map<Integer, SimpleMatrix> entity2vec = new HashMap<>();
	public static Map<Integer, SimpleMatrix> relation2vec = new HashMap<>();
	public static HashMap<Integer, SimpleMatrix> matrix_wr = new HashMap<Integer, SimpleMatrix>();
	public static HashMap<Integer, SimpleMatrix> matrix_r = new HashMap<Integer, SimpleMatrix>();
	public static HashMap<Integer, SimpleMatrix> matrix_e = new HashMap<Integer, SimpleMatrix>();
	public static SimpleMatrix matrix_I = new SimpleMatrix(m, m);
	public static Map<Integer, Double> threshold = new HashMap<>();
	public static HashMap<String, HashSet<Integer>> ex1 = new HashMap<>();
	public static HashMap<String, HashSet<Integer>> ex2 = new HashMap<>();
	public static HashSet<Integer> ent = new HashSet<>();
	public static HashSet<Integer> tail = new HashSet<>();

	public static void main(String[] args) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter("log.txt"));
		BufferedWriter bw1 = new BufferedWriter(new FileWriter("result1.txt"));
		int total = 0;
		prepare();
		File srcFile = new File("data/FB15k/test.txt");
		BufferedReader br = new BufferedReader(new FileReader(srcFile));
		String line;
		ArrayList<String> test = new ArrayList<>();
		while ((line = br.readLine()) != null) {
			test.add(line);
		}
		for (String s : test) {
			HashMap<Integer, Double> score = new HashMap<>();
			String[] tmp = s.split("\t");
			String hRel = tmp[HEAD_IDX] + "\t" + tmp[RELATION_IDX];
			String tRel = tmp[TAIL_IDX] + "\t" + tmp[RELATION_IDX];
			int h = entity2id.get(tmp[HEAD_IDX]);
			int r = relation2id.get(tmp[RELATION_IDX]);
			int t = entity2id.get(tmp[TAIL_IDX]);
			for (int tt = 0; tt < entity2id.size(); tt++) {
				score.put(tt, calcScore(h, tt, r));
			}
			sortByValue(hRel, score, t, 1, bw, total);

			for (int hh = 0; hh < entity2id.size(); hh++) {
				score.put(hh, calcScore(hh, t, r));
			}
			sortByValue(tRel, score, h, -1, bw, total);

			total++;
			if (total % 1 == 0) {
				System.out.println(total + ":" + RIGHT_NUM * 1.0 / (total * 2) + "-" + RANK_NUM * 1.0 / (total * 2)
						+ "**" + F_RIGHT_NUM * 1.0 / (total * 2) + "-" + F_RANK_NUM * 1.0 / (total * 2));
			}
			bw1.write(total + ":" + RIGHT_NUM * 1.0 / (total * 2) + "-" + RANK_NUM * 1.0 / (total * 2)
						+ "**" + F_RIGHT_NUM * 1.0 / (total * 2) + "-" + F_RANK_NUM * 1.0 / (total * 2));
			bw1.newLine();
			bw1.flush();
		}

		br.close();
		bw.close();
		bw1.close();
	}

	// ����map��ֵ��������
	public static void sortByValue(String str, HashMap<Integer, Double> score, int t, int flag, BufferedWriter bw,
			int total) throws IOException {
		HashSet<Integer> tt = new HashSet<>();
		if (flag == 1) {
			tt = ex1.get(str);
		} else {
			tt = ex2.get(str);
		}
		List<Map.Entry<Integer, Double>> list = new ArrayList<Map.Entry<Integer, Double>>(score.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			@Override
			// ��С����
			public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
				if (o2.getValue() > o1.getValue()) {
					return -1;
				}
				if (o2.getValue() < o1.getValue()) {
					return 1;
				}
				return 0;
			}
		});
		int count = 0;
		int filt_count = 0;
		for (Entry<Integer, Double> mapping : list) {
			int ent = mapping.getKey();
			if (count < 10) {
				if (t == ent) {
					RIGHT_NUM++;
				}
			}
			if (t == ent) {
				RANK_NUM += (count + 1);
				if (count > 10000) {
					bw.write((total + 1) + "---" + str + "-----" + (count + 1));
					bw.newLine();
					bw.flush();
					System.out.println("--------------" + str + "-----" + (count + 1));
				}
			}
			count++;
			if (filt_count < 10) {
				if (t == ent) {
					F_RIGHT_NUM++;
				}
			}
			if (t == ent) {
				F_RANK_NUM += (filt_count + 1);
				break;
			}
			if (!tt.contains(ent)) {
				filt_count++;
			}
		}
	}

	public static void prepare() throws IOException {
		String srcPath = "data/FB15k/";
		String dataPath = "MSP/";
		String line;
		int count = 0;
		BufferedReader br1 = new BufferedReader(new FileReader(srcPath + "entity2id.txt"));
		BufferedReader br2 = new BufferedReader(new FileReader(srcPath + "relation2id.txt"));
		BufferedReader br3 = new BufferedReader(new FileReader(dataPath + "entity2vec.bern"));
		BufferedReader br4 = new BufferedReader(new FileReader(dataPath + "relation2vec.bern"));
		BufferedReader br5 = new BufferedReader(new FileReader(dataPath + "matrix_wr.bern"));
		BufferedReader br6 = new BufferedReader(new FileReader(dataPath + "matrix_r.bern"));
		BufferedReader br7 = new BufferedReader(new FileReader(dataPath + "matrix_e.bern"));

		while ((line = br1.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			entity2id.put(tokens[0], new Integer(tokens[1]));
			entity_num++;
		}

		while ((line = br2.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			relation2id.put(tokens[0], new Integer(tokens[1]));
			relation_num++;
		}

		double[] data;
		while ((line = br3.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			data = new double[m];
			for (int i = 0; i < m; i++) {
				data[i] = new Double(tokens[i]);
			}
			entity2vec.put(count, new SimpleMatrix(m, 1, true, data));
			count++;
		}

		count = 0;
		while ((line = br4.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			data = new double[m];
			for (int i = 0; i < m; i++) {
				data[i] = new Double(tokens[i]);
			}
			relation2vec.put(count, new SimpleMatrix(m, 1, true, data));
			count++;
		}

		count = 0;
		while ((line = br5.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			data = new double[m];
			for (int i = 0; i < m; i++) {
				data[i] = new Double(tokens[i]);
			}
			matrix_wr.put(count, new SimpleMatrix(m, 1, true, data));
			count++;
		}

		count = 0;
		while ((line = br6.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			data = new double[m];
			for (int i = 0; i < m; i++) {
				data[i] = new Double(tokens[i]);
			}
			matrix_r.put(count, new SimpleMatrix(m, 1, true, data));
			count++;
		}

		count = 0;
		while ((line = br7.readLine()) != null) {
			String[] tokens = line.split("\\s+");
			data = new double[m];
			for (int i = 0; i < m; i++) {
				data[i] = new Double(tokens[i]);
			}
			matrix_e.put(count, new SimpleMatrix(1, m, true, data));
			count++;
		}

		br1.close();
		br2.close();
		br3.close();
		br4.close();
		br5.close();
		br6.close();
		br7.close();

		for (int ii = 0; ii < m; ii++) {
			for (int jj = 0; jj < m; jj++) {
				if (ii == jj) {
					matrix_I.set(ii, jj, 1);
				}
			}
		}

		File file1 = new File(srcPath + "train.txt");
		BufferedReader br = new BufferedReader(new FileReader(file1));
		String headName, tailName, relName;
		while ((line = br.readLine()) != null) {
			String[] tmp = line.split("\t");
			headName = tmp[HEAD_IDX];
			tailName = tmp[TAIL_IDX];
			relName = tmp[RELATION_IDX];
			HashSet<Integer> tmp_t;
			String str_h = headName + "\t" + relName;
			String str_t = tailName + "\t" + relName;
			if (ex1.get(str_h) == null) {
				tmp_t = new HashSet<>();
			} else {
				tmp_t = ex1.get(str_h);
			}
			tmp_t.add(entity2id.get(tailName));
			ex1.put(str_h, tmp_t);

			if (ex2.get(str_t) == null) {
				tmp_t = new HashSet<>();
			} else {
				tmp_t = ex2.get(str_t);
			}
			tmp_t.add(entity2id.get(headName));
			ex2.put(str_t, tmp_t);
		}
		br.close();

		File file2 = new File(srcPath + "valid.txt");
		br = new BufferedReader(new FileReader(file2));
		while ((line = br.readLine()) != null) {
			String[] tmp = line.split("\t");
			headName = tmp[HEAD_IDX];
			tailName = tmp[TAIL_IDX];
			relName = tmp[RELATION_IDX];
			HashSet<Integer> tmp_t;
			String str_h = headName + "\t" + relName;
			String str_t = tailName + "\t" + relName;
			if (ex1.get(str_h) == null) {
				tmp_t = new HashSet<>();
			} else {
				tmp_t = ex1.get(str_h);
			}
			tmp_t.add(entity2id.get(tailName));
			ex1.put(str_h, tmp_t);

			if (ex2.get(str_t) == null) {
				tmp_t = new HashSet<>();
			} else {
				tmp_t = ex2.get(str_t);
			}
			tmp_t.add(entity2id.get(headName));
			ex2.put(str_t, tmp_t);
		}
		br.close();

		File file3 = new File(srcPath + "test.txt");
		br = new BufferedReader(new FileReader(file3));
		while ((line = br.readLine()) != null) {
			String[] tmp = line.split("\t");
			headName = tmp[HEAD_IDX];
			tailName = tmp[TAIL_IDX];
			relName = tmp[RELATION_IDX];
			HashSet<Integer> tmp_t;
			String str_h = headName + "\t" + relName;
			String str_t = tailName + "\t" + relName;
			if (ex1.get(str_h) == null) {
				tmp_t = new HashSet<>();
			} else {
				tmp_t = ex1.get(str_h);
			}
			tmp_t.add(entity2id.get(tailName));
			ex1.put(str_h, tmp_t);

			if (ex2.get(str_t) == null) {
				tmp_t = new HashSet<>();
			} else {
				tmp_t = ex2.get(str_t);
			}
			tmp_t.add(entity2id.get(headName));
			ex2.put(str_t, tmp_t);
		}
		br.close();
	}

	public static double calcScore(int eh, int et, int rel) {
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
}