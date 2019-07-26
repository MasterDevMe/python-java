import java.io.*;
import java.net.Socket;
import java.sql.*;
import java.util.Date;
import java.util.*;
class Main {

 private Object deepCopy(Object object) {
   try {
     ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
     ObjectOutputStream outputStrm = new ObjectOutputStream(outputStream);
     outputStrm.writeObject(object);
     ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
     ObjectInputStream objInputStream = new ObjectInputStream(inputStream);
     return objInputStream.readObject();
   }
   catch (Exception e) {
     e.printStackTrace();
     return null;
   }
 }

  public Vector retrieve_data(float time_window, boolean disp) throws Exception{
      Socket datasocket = new Socket("http://localhost", 8000);

        // self.datasocket.connect("tcp://localhost:{}".format(self.dataport))
        // self.datasocket.setsockopt(zmq.SUBSCRIBE, "10000")
        // #flush cache
        Date date= new Date();      
        long t1 = date.getTime();
        String d = "";

        while((new Date()).getTime()-t1 < 0.1) {
            t1 = date.getTime();
        }

        ArrayList TS = new ArrayList<Float>();
        ArrayList S1 = new ArrayList<Float>();
        ArrayList S1OS = new ArrayList<Float>();
        ArrayList S2 = new ArrayList<Float>();
        ArrayList S2OS = new ArrayList<Float>();
        ArrayList TEMP = new ArrayList<Float>();

        // , S1, S1OS, S2, S2OS, TEMP = new ArrayList<Float>();
        while(true){
            if( disp ){
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
            if((t-t1) > time_window)
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
    int   n = 5;
    float nsig = 3.0f;

    for(int i = 0 ; i < S1.size() ; i ++) {
        if (i < n/2) {
          continue;          
        } else if ( i >= S1.size() - n/2) {
          continue;
        } else {
            List<Float> s = S1.subList(i, i + n); 
            // s = list(S1[i:i+n])
            float val = s.get(n/2);
            float sum = 0.0f;
             // m = np.mean(s)
            for(int j = 0 ; j < s.size() ; j ++) {
              sum += s.get(j);
            }
            float m = sum / s.size();
            // std = np.std(s)

            sum = 0.0f;
            for(int j = 0 ; j < s.size() ; j ++) {
              sum += Math.pow(s.get(j) - m, 2);
            }
            float std = (float)(Math.sqrt((float)(sum / s.size())));

            if (Math.abs(val - m) > nsig*std) {
              So.add(i, m);
            } else {
              So.add(i, val);
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
    float B = b * k + a * (m - s2) - c * g - f*(d-s1);
    // A = a*k - c*f
    // B = b*k + a*(m-s2) - c*g - f*(d-s1)
    float C = b*(m-s2) - g*(d-s1);

    ArrayList res = new ArrayList<Float>();

    if (Math.pow(B, 2) - 4.0*A*C > 0.0) {
        float val1 = -B/(2*A) + (float)Math.sqrt(Math.pow(B, 2) - 4.0*A*C)/(2*A); 
        float val2 = -B/(2*A) - (float)Math.sqrt(Math.pow(B, 2) - 4.0*A*C)/(2*A);
        res.add(val1);
        res.add(val2);
        return res;
    }
    return res;
  }

  public static void main(String[] args) {
    System.out.println("Hello world!");
  }
}