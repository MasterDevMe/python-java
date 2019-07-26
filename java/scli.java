import java.io.*;
import java.net.Socket;
import java.sql.*;
import java.util.Date;
import java.util.*;
class test {
  private Map<String, ArrayList<Float>> caldata;
  public test() {
  }

  public Vector retrieve_data(float time_window, boolean disp) throws Exception {
    Socket datasocket = new Socket("http://localhost", 8000);

    // self.datasocket.connect("tcp://localhost:{}".format(self.dataport))
    // self.datasocket.setsockopt(zmq.SUBSCRIBE, "10000")
    // #flush cache
    Date date = new Date();
    long t1 = date.getTime();
    String d = "";

    while ((new Date()).getTime() - t1 < 0.1) {
      t1 = date.getTime();
    }

    ArrayList TS = new ArrayList<Float>();
    ArrayList S1 = new ArrayList<Float>();
    ArrayList S1OS = new ArrayList<Float>();
    ArrayList S2 = new ArrayList<Float>();
    ArrayList S2OS = new ArrayList<Float>();
    ArrayList TEMP = new ArrayList<Float>();

    // , S1, S1OS, S2, S2OS, TEMP = new ArrayList<Float>();
    while (true) {
      if (disp) {
        System.out.println(".");
        System.out.flush();

      }
      BufferedReader reader = new BufferedReader(new InputStreamReader(datasocket.getInputStream()));
      String line;
      while ((line = reader.readLine()) != null)
        d += line + "\n";

      String data = d.split(" ")[1];

      String ts = data.split(",")[0];
      String s1 = data.split(",")[1];
      String s1offset = data.split(",")[2];
      String s2 = data.split(",")[3];
      String s2offset = data.split(",")[4];
      String temp = data.split(",")[5];

      TS.add(Float.valueOf(ts));
      S1.add(Float.parseFloat(s1));
      S1OS.add(Float.parseFloat(s1offset));
      S2.add(Float.parseFloat(s2));
      S2OS.add(Float.parseFloat(s2offset));
      TEMP.add(Float.parseFloat(temp));
      float t = Float.valueOf(ts);
      if ((t - t1) > time_window)
        break;
    }
    System.out.println();
    // self.datasocket.disconnect("tcp://localhost:{}".format(self.dataport));
    // ArrayList<ArrayList
    Vector result_array = new Vector<ArrayList>();
    result_array.add(TS);
    result_array.add(S1);
    result_array.add(S1OS);
    result_array.add(S2);
    result_array.add(S2OS);
    result_array.add(TEMP);
    datasocket.close();
    // float[][] arrays = new float[][] { TS, S1, S1OS, S2, S2OS, TEMP };
    return result_array;

  }

  public ArrayList<Float> removeOutliers(ArrayList<Float> S1) {
    ArrayList<Float> So = S1;
    // this.deepCopy(S1.toArray());
  
    int n = 5;
    float nsig = 3.0f;
    System.out.println(S1.size());
    for (int i = 0; i < S1.size(); i++) {
      if (i < n / 2) {
        continue;
      } else if (i >= S1.size() - n / 2) {
        continue;
      } else {
        List<Float> s;
        if ( i+n >= S1.size()) {
          s = S1.subList(i, S1.size() - 1);
        } else {
          s = S1.subList(i, i + n);
        }

        // System.out.println(s);
        // s = list(S1[i:i+n])
        float val = 0.0f;
        if( n/2 >= s.size()) {
          val = s.get(s.size() - 1); 
        } else {
          val = s.get(n / 2);
        }

        float sum = 0.0f;
        // m = np.mean(s)
        for (int j = 0; j < s.size(); j++) {
          sum += s.get(j);
        }
        float m = sum / s.size();
        // std = np.std(s)

        sum = 0.0f;
        for (int j = 0; j < s.size(); j++) {
          sum += Math.pow(s.get(j) - m, 2);
        }
        float std = (float) (Math.sqrt((float) (sum / s.size())));

        if (Math.abs(val - m) > nsig * std) {
          So.set(i, m);
        } else {
          So.set(i, val);
        }
      }
    }
    return So;
  }

  public ArrayList<Float> solve(float s1, float s2, ArrayList<Float> coefs) {
    // a, b, c, d, f, g, k, m = coefs
    float a = coefs.get(0);
    float b = coefs.get(1);
    float c = coefs.get(2);
    float d = coefs.get(3);
    float f = coefs.get(4);
    float g = coefs.get(5);
    float k = coefs.get(6);
    float m = coefs.get(7);
    float A = a * k - c * f;
    float B = b * k + a * (m - s2) - c * g - f * (d - s1);
    // A = a*k - c*f
    // B = b*k + a*(m-s2) - c*g - f*(d-s1)
    float C = b * (m - s2) - g * (d - s1);

    ArrayList res = new ArrayList<Float>();

    if (Math.pow(B, 2) - 4.0 * A * C > 0.0) {
      float val1 = -B / (2 * A) + (float) Math.sqrt(Math.pow(B, 2) - 4.0 * A * C) / (2 * A);
      float val2 = -B / (2 * A) - (float) Math.sqrt(Math.pow(B, 2) - 4.0 * A * C) / (2 * A);
      res.add(val1);
      res.add(val2);
      return res;
    }
    return res;
  }

  public void continouse_weight(float twin, float inttime) throws Exception {
    if(twin == -1.0f) 
      twin = 10.0f;   
    if(inttime == -1.0f)
      inttime = 2.0f;

    ArrayList<Float> coefs = new ArrayList<Float>();

    try {
      coefs = this.caldata.get("cal");
    }catch (NullPointerException ex) {
      System.err.println("ERROR: Calibration not updated...");
      return;
    }


    System.out.print("Set Weight Refrence and <Enter> (q to quit)");
    char res = (char) System.in.read();
    if (res == 'q') {
      return;
    }


    Vector<ArrayList<Float>> arr_retrieve_data = this.retrieve_data(inttime, true);
    ArrayList<Float> ts = (ArrayList)arr_retrieve_data.get(0);
    ArrayList<Float> s1 = (ArrayList)arr_retrieve_data.get(1);
    ArrayList<Float> s1o = (ArrayList)arr_retrieve_data.get(2);
    ArrayList<Float> s2 = (ArrayList)arr_retrieve_data.get(3);
    ArrayList<Float> s2o = (ArrayList)arr_retrieve_data.get(4);
    ArrayList<Float> temp = (ArrayList)arr_retrieve_data.get(5);

    ArrayList<Float> removeResult1 = this.removeOutliers(s1);
    ArrayList<Float> removeResult2 = this.removeOutliers(s2);
    float sum1 = 0.0f, sum2 = 0.0f;
    for(int i =0 ; i<removeResult1.size(); i ++) {
      sum1 += removeResult1.get(i);
    }

    for(int i =0 ; i<removeResult2.size(); i ++) {
      sum2 += removeResult2.get(i);
    }

    float s1ref = sum1 / removeResult1.size();
    float s2ref = sum2 / removeResult2.size();

    
    System.out.print("Ready? <Enter> (q to quit)");
    res = (char) System.in.read();
    if (res == 'q') {
      return;
    }

    Date date = new Date();
    long t1 = date.getTime();
    long dt = date.getTime() - t1;

    while (dt < twin) {
      arr_retrieve_data = this.retrieve_data(inttime, false);
      ts = (ArrayList)arr_retrieve_data.get(0);
      s1 = (ArrayList)arr_retrieve_data.get(1);
      s1o = (ArrayList)arr_retrieve_data.get(2);
      s2 = (ArrayList)arr_retrieve_data.get(3);
      s2o = (ArrayList)arr_retrieve_data.get(4);
      temp = (ArrayList)arr_retrieve_data.get(5);

      sum1 = 0.0f; sum2 = 0.0f;
      removeResult1 = this.removeOutliers(s1);
      removeResult2 = this.removeOutliers(s2);
      
      for(int i =0 ; i<removeResult1.size(); i ++) {
        sum1 += removeResult1.get(i);
      }
  
      for(int i =0 ; i<removeResult2.size(); i ++) {
        sum2 += removeResult2.get(i);
      }

      float ds1 = sum1 / removeResult1.size() - s1ref;
      float ds2 = sum2 / removeResult2.size() - s2ref;

      ArrayList<Float> solveResult = this.solve(ds1, ds2, coefs);
      float wt = Math.max(solveResult.get(0), solveResult.get(1));
      System.out.println("\033[F");
      System.out.println("\033[K");
      System.out.println("Weight: {" +  Float.toString(wt) + "} g.");
      dt = date.getTime()-t1;
    }
    // res = System.in;

  } 
}
public class scli {
  public static void main(String[] args) {
    System.out.println("Hello world!");
    test test1 = new test();
    ArrayList res = new ArrayList<Float>();
    res.add((float)3.0);
    res.add((float)4.0);
    res.add((float)9.0);
    res.add((float)6.0);
    res.add((float)90.0);
    res.add((float)8.0);
    res.add((float)79.0);
    res.add((float)15.0);
    // System.out.println(test1.solve((float)1.0, (float)2.0, res));
    System.out.println(test1.removeOutliers(res));
    try {
		  test1.continouse_weight(-1.0f, -1.0f);
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
}
