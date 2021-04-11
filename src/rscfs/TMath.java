/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package rscfs;
//import weka.core.Instances;
import java.util.Random;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.inference.WilcoxonSignedRankTest;



/**
 *
 * @author truongtran
 */
public class TMath
{
    /**
     *Caculate Mean
     */
    public static double Mean(double[] data)
    {
        double sum=0;
        for(int i=0;i<data.length;i++)
            sum+=data[i];
        return sum/data.length;
    }
    /**
     *Caculate Mean of different between two arrays
     */
    public static double MeanDif(double[] data1,double[] data2)
    {
        double sum=0;
        for(int i=0;i<data1.length;i++)
            sum+=data1[i]-data2[i];
        return sum/data1.length;
    }
    /**
     * Caculate standard deviation
     * @param data
     * @return
     */
    public static double Stdv(double[] data)
    {
        double mean=Mean(data);
        double sum=0;
        for(int i=0;i<data.length;i++)
            sum+=Math.pow(data[i]-mean, 2);
        sum=Math.sqrt(sum/data.length);
        return sum;

    }
    /**
     * Caculate standard deviation
     * @param data
     * @return
     */
    public static double Stdv(double[] data,int numFolds)
    {
        double mean=Mean(data);
        double sum=0;
        double subSum;
        int iteration=data.length/numFolds;
        for(int i=0;i<data.length;i=i+numFolds)
        {
            subSum=0;
            for(int j=i;j<i+numFolds;j++)
                subSum+=data[j];
            subSum/=numFolds;
            sum+=Math.pow(subSum-mean, 2);
        }
        sum=Math.sqrt(sum/iteration);
        return sum;

    }
    public static double StdvDf(double[] data1,double[] data2)
    {
        double mean=MeanDif(data1,data2);
        double sum=0;
        for(int i=0;i<data1.length;i++)
            sum+=Math.pow(data1[i]-data2[i]-mean, 2);
        sum=Math.sqrt(sum/data1.length);
        return sum;

    }
    public static String pValue(double[] data1,double[] data2, int numFolds)
    {
        double dt1[]=new double[data1.length/numFolds];
        double dt2[]=new double[data1.length/numFolds];
        for(int i=0;i<data1.length/numFolds;i++)
        {
            for(int j=0;j<numFolds;j++)
            {
                dt1[i]+=data1[i*numFolds+j];
                dt2[i]+=data2[i*numFolds+j];
            }
            dt1[i]/=numFolds;
            dt2[i]/=numFolds;
        }
        TTest t=new TTest();
        double pvalue=t.pairedTTest(dt1, dt2);
        String spvalue=Double.toString(pvalue);    
        if(pvalue>0.05)
            spvalue+="=";
        else
            if(pvalue>0.01)
                spvalue+="*";
            else
                if(pvalue>0.001)
                    spvalue+="**";
                else
                    spvalue+="***";
        return spvalue;
    }
    
    public static String pWinValue(double[] data1,double[] data2)
    {
        WilcoxonSignedRankTest wilTest=new WilcoxonSignedRankTest();
        double pvalue=wilTest.wilcoxonSignedRankTest(data1, data2, true);
        String spvalue=Double.toString(pvalue);    
        if(pvalue>0.05)
            spvalue+="=";
        else
            if(pvalue>0.01)
                spvalue+="*";
            else
                if(pvalue>0.001)
                    spvalue+="**";
                else
                    spvalue+="***";
        return spvalue;
    }
    public static String pWinValue(double[] data1,double[] data2, int numFolds)
    {
        double dt1[]=new double[data1.length/numFolds];
        double dt2[]=new double[data1.length/numFolds];
        for(int i=0;i<data1.length/numFolds;i++)
        {
            for(int j=0;j<numFolds;j++)
            {
                dt1[i]+=data1[i*numFolds+j];
                dt2[i]+=data2[i*numFolds+j];
            }
            dt1[i]/=numFolds;
            dt2[i]/=numFolds;
        }
        WilcoxonSignedRankTest wilTest=new WilcoxonSignedRankTest();
        double pvalue=wilTest.wilcoxonSignedRankTest(dt1, dt2, true);
        //String spvalue=Double.toString(pvalue);    
        String spvalue="";
        if(pvalue>0.05)
            spvalue+="=";
        else
            if(pvalue>0.01)
                spvalue+="*";
            else
                if(pvalue>0.001)
                    spvalue+="**";
                else
                    spvalue+="***";
        return spvalue;
    }
    public static double tValue(double[] data1,double[] data2, int numFolds)
    {
        double dt1[]=new double[data1.length/numFolds];
        double dt2[]=new double[data1.length/numFolds];
        for(int i=0;i<data1.length/numFolds;i++)
        {
            for(int j=0;j<numFolds;j++)
            {
                dt1[i]+=data1[i*numFolds+j];
                dt2[i]+=data2[i*numFolds+j];
            }
            dt1[i]/=numFolds;
            dt2[i]/=numFolds;
        }
        TTest t=new TTest();
        //OneWayAnova anova=new OneWayAnova();
        double tvalue=t.tTest(dt1, dt2);
        return tvalue;
    }
    public static String toString(double d)
    {
        if(Double.toString(d).length()<=3)
            return Double.toString(d)+"00";
        else
            if(Double.toString(d).length()==4)
                return Double.toString(d)+"0";
            else
                return Double.toString(d).substring(0, 5);
    }
    

}
