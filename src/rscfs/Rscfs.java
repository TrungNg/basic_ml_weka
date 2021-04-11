/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rscfs;
import java.io.IOException;
import java.util.Random;
import java.util.Arrays;
import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author truongtran
 */
public class Rscfs
{
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception
    {
        // TODO code application logic here
//        String fileName="zoo";
//        String path ="../datasets/uci_zoo/";
//        Instances data = DatasetHandle.ReadDataARFF(path+fileName+".arff");
//        data = DatasetHandle.RemoveAttributes(data, new int[]{0});
//        data.setClassIndex(data.numAttributes() - 1);
        /*---------------------------------------------*/
//        String fileName="SPECT.data";
//        String path ="../datasets/uci_SPECT_heart/";
//        Instances data = DatasetHandle.ReadDataCSV(path+fileName);
//        data.setClassIndex(0);
        /*---------------------------------------------*/
//        String fileName="Training Dataset.arff";
//        String path ="../datasets/uci_phishing_websites/";
//        Instances data = DatasetHandle.ReadDataARFF(path+fileName);
//        data.setClassIndex(data.numAttributes() - 1);
        /*---------------------------------------------*/
//        String fileName="phpp41X7N.arff";
//        String path ="../datasets/webdata_wXa/";
//        Instances data = DatasetHandle.ReadDataARFF(path+fileName);
//        data.setClassIndex(0);
        /*---------------------------------------------*/
        String fileName="php2BsQch.arff";
        String path ="../datasets/connect-4/";
        Instances orgData = DatasetHandle.ReadDataARFF(fileName);
        NumericToNominal converter = new NumericToNominal();
    	String[] filterOptions = new String[2];
    	filterOptions[0] = "-R";
    	filterOptions[1] = "first-last";
        converter.setOptions(filterOptions);
        converter.setInputFormat(orgData);
        Instances data = Filter.useFilter(orgData, converter);
        data.setClassIndex(data.numAttributes() - 1);


        int[] rseeds = new int[] {22246426,212603442,83029677,106630963,57111782,112265964,223969767,
        		68633838,1203679,62725478,178009373,139933502,258126040,164936307,98399639,212884694,
        		114594252,89916786,209254548,221938410,76123412,150333749,58129987,148057717,41451171,
        		19787546,217330000,118313133,130536775,238934855};
        double[] mlpAcc = new double[30];
        double[] j48Acc = new double[30];
        double[] rforestAcc = new double[30];
        double nbAcc = 0;
        double svmAcc = 0;

        if(data.numInstances() < 1000) {
            nbAcc = classifier.NBCV(data); // no change with different seeds
            svmAcc = classifier.SMOCV(data); // no change with different seeds?
        	for(int i=0;i <30;i++) {
        		mlpAcc[i] = classifier.MLPCV(rseeds[i], data);
        		j48Acc[i] = classifier.J48CV(rseeds[i], data);
        		rforestAcc[i] = classifier.RandomForestCV(rseeds[i], data,200);
            }
        }
        else {
        	data.randomize(new Random(0));
        	Instances[] splits = DatasetHandle.splitData(data,10,7);
        	Instances trainData = splits[0];
        	Instances testData = splits[1];
        	nbAcc = classifier.NB(trainData, testData); // no change with different seeds
        	svmAcc = classifier.SMO(trainData, testData); // no change with different seeds?
        	for(int i=0;i <10;i++) {
        		mlpAcc[i] = classifier.MLP(rseeds[i], trainData, testData);
        		j48Acc[i] = classifier.J48(rseeds[i], trainData, testData);
        		rforestAcc[i] = classifier.RandomForest(rseeds[i], trainData, testData, 200);
        	}
        }
        System.out.println("---\nFinal Results: ");
        System.out.println("NB: " + Double.toString(nbAcc));
        System.out.println("SVM: " + Double.toString(svmAcc));
        System.out.println("MLP: mean " + Double.toString(TMath.Mean(mlpAcc)) + ", dev " + Double.toString(TMath.Stdv(mlpAcc)));
        System.out.println(Arrays.toString(mlpAcc));
        System.out.println("J48: mean " + Double.toString(TMath.Mean(j48Acc)) + ", dev " + Double.toString(TMath.Stdv(j48Acc)));
        System.out.println(Arrays.toString(j48Acc));
        System.out.println("RandomForest: mean " + Double.toString(TMath.Mean(rforestAcc)) + ", dev " + Double.toString(TMath.Stdv(rforestAcc)));
        System.out.println(Arrays.toString(rforestAcc));
    }

    // This function is equivalent to using crossValidateModel
    public static void crossValidation(Instances data, int numFolds) throws Exception
    {
        // The below commented part is equivalent to using MLPCV (same result if using same Random seed)
    	int seed = 0;
        Instances[] trainFolds = new Instances[10];
        Instances[] testFolds = new Instances[10];
        Instances[] splitted_sets = new Instances[20];
        splitted_sets = DatasetHandle.splitFolds(data, numFolds, new Random(0));
        //testFolds = DatasetHandle.testFolds(data, numFolds);
        trainFolds = Arrays.copyOfRange(splitted_sets,0,10);
        testFolds = Arrays.copyOfRange(splitted_sets,10,20);
        double[] results = new double[numFolds];
        double[] weighted_res = new double[numFolds];
        for(int i=0;i<numFolds;i++){
            results[i] = classifier.MLP(seed, trainFolds[i], testFolds[i]);
            weighted_res[i] = results[i] * testFolds[i].size() / ((double)data.size()/numFolds);
            System.out.println(results[i]);
        }
        System.out.println("Stats of first 10 folds:");
        System.out.println(TMath.Mean(weighted_res));
        System.out.println(TMath.Stdv(weighted_res));
    }
}
