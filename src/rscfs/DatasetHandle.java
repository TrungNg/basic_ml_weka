    /*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package rscfs;

import java.io.*;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.InstanceComparator;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.RemoveFolds;

/**
 *
 * @author truongtran
 */
public class DatasetHandle
{
    //read a dataset from file
    public static Instances ReadDataARFF(String path) throws IOException
    {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();
        return data;
    }
    //read a dataset from csv file
    public static Instances ReadDataCSV(String path) throws IOException, Exception
    {
    	CSVLoader loader = new CSVLoader();
    	NumericToNominal converter = new NumericToNominal();
    	String[] options = new String[2];
    	options[0] = "-R";
    	options[1] = "1-2";

        loader.setSource(new File(path));
        Instances data = loader.getDataSet();

        converter.setOptions(options);
        converter.setInputFormat(data);

        Instances converted_data = Filter.useFilter(data, converter);
        converted_data.setClassIndex(0);

        return converted_data;
    }
    //Remove an attribute from dataset
    public static Instances RemoveAttributes(Instances orInsts, int[] indices) throws Exception
    {
    	Remove remover = new Remove();
    	Instances temInsts = new Instances(orInsts);
    	remover.setAttributeIndicesArray(indices);
    	//remover.setInvertSelection(true);			// use this if indices are attributes to be kept
    	remover.setInputFormat(temInsts);
        return Filter.useFilter(temInsts, remover);
    }
    //standardise data
    public static Instances Standardise(Instances orInsts) throws Exception
    {
        Instances temInsts = new Instances(orInsts);
        Standardize std = new Standardize();
        std.setInputFormat(temInsts);
        return Filter.useFilter(temInsts, std);
    }
    //discretise data
    public static Instances Discretize(Instances orInsts) throws Exception
    {
        Instances temInsts = new Instances(orInsts);
        Discretize dis = new Discretize();
        dis.setBins(3);
        dis.setInputFormat(temInsts);
        return Filter.useFilter(temInsts, dis);
    }
    //get 10 training folds cross-validation
    public static Instances[] trainFolds(Instances orInsts, int numFolds)
    {
        Instances[] folds = new Instances[numFolds];
        //Random rdg = new Random();
        for(int i=0;i<numFolds;i++)
            folds[i]=new Instances(orInsts.trainCV(numFolds,i));
        return folds;
    }
    //get 10 testing folds cross-validation
    public static Instances[] testFolds(Instances orInsts, int numFolds)
    {
        Instances[] folds = new Instances[numFolds];
        for(int i=0;i<numFolds;i++)
            folds[i]=new Instances(orInsts.testCV(numFolds, i));
        return folds;
    }
    //get 10 training folds cross-validation with random generator
    public static Instances[] trainFolds(Instances orInsts, int numFolds, Random rnd)
    {
        Instances[] folds = new Instances[numFolds];
        //Random rdg = new Random();
        for(int i=0;i<numFolds;i++)
            folds[i]=new Instances(orInsts.trainCV(numFolds,i,rnd));
        return folds;
    }
    //get 10 training folds cross-validation with random generator
    public static Instances[] splitFolds(Instances orInsts, int numFolds, Random rnd)
    {
    	Instances allInsts = new Instances(orInsts);
    	allInsts.randomize(rnd);
    	if(allInsts.classAttribute().isNominal())
    		allInsts.stratify(numFolds);
        Instances[] folds = new Instances[2*numFolds];
        //Random rdg = new Random();
        for(int i=0;i<numFolds;i++) {
            folds[i]=new Instances(allInsts.trainCV(numFolds,i,rnd));
            folds[i+numFolds]=new Instances(allInsts.testCV(numFolds,i));
        }
        return folds;
    }
    private boolean isExist(Instances insts, Instance inst)
    {
    	for(int i=0;i<insts.numInstances();i++)
    	{
    		boolean check= true;
    		for(int j=0;j<inst.numAttributes();j++)
    			if(insts.instance(i).value(j)!=inst.value(j))
    				check=false;
    		if(check)
    			return true;
    	}
    	return false;
    }
    // Split data using StratifiedRemoveFolds, s
    public static Instances[] splitData(Instances allData, int numFolds, int trainFolds) throws Exception
    {
    	StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

    	// set options for creating the subset of data
    	String[] options = new String[4];

    	options[0] = "-N";                 // indicate we want to set the number of folds
    	options[1] = Integer.toString(numFolds);  // split the data into five random folds
    	options[2] = "-F";                 // indicate we want to select a specific fold
    	options[3] = Integer.toString(trainFolds+1);  // select the first fold
//    	options[4] = "-S";                 // indicate we want to set the random seed
//    	options[5] = Integer.toString(0);  // set the random seed to 1

    	filter.setOptions(options);        // set the filter options
    	filter.setInputFormat(allData);       // prepare the filter for the data format
    	filter.setInvertSelection(false);  // do not invert the selection (default 0, no randomizing)

    	// apply filter for test data here
    	Instances test = Filter.useFilter(allData, filter);
    	Instances train = new Instances(test,0);

    	for( int i=1;i<=10;i++ ) {
    		if(i==trainFolds+1) {
    			continue;
    		}
        	RemoveFolds subfilter = new RemoveFolds();

        	// set options for creating the subset of data
        	String[] suboptions = new String[4];

//        	suboptions[0] = "-V";                 // indicate we want to set the number of folds
//        	suboptions[1] = Boolean.toString(false);
        	suboptions[0] = "-N";                 // indicate we want to set the number of folds
        	suboptions[1] = Integer.toString(numFolds);  // split the data into five random folds
        	suboptions[2] = "-F";                 // indicate we want to select a specific fold
        	suboptions[3] = Integer.toString(trainFolds+1);  // select the first fold

        	subfilter.setOptions(suboptions);        // set the filter options
        	subfilter.setInputFormat(allData);       // prepare the filter for the data format
        	subfilter.setInvertSelection(false);  // do not invert the selection

        	// apply filter for test data here
        	Instances fold = Filter.useFilter(allData, subfilter);
        	if(i<=trainFolds) {
    			for(int j=0; j<fold.numInstances(); j++) {
    				train.add(fold.get(j));
    			}
        	}
        	else{
    			for(int k=0; k<fold.numInstances(); k++) {
    				test.add(fold.get(k));
    			}
        	}
    	}
    	Instances[] res = new Instances[] {train, test};
    	return res;
    }
    // Split data using RemoveFolds, all instances are kept the same order, no randomize, no stratify
    public static Instances[] splitDataNoStratify(Instances allData, int testFold) throws Exception
    {
    	RemoveFolds filter = new RemoveFolds();

    	// set options for creating the subset of data
    	String[] options = new String[4];

    	options[0] = "-N";                 // indicate we want to set the number of folds
    	options[1] = Integer.toString(10);  // split the data into five random folds
    	options[2] = "-F";                 // indicate we want to select a specific fold
    	options[3] = Integer.toString(testFold);  // select the first fold

    	filter.setOptions(options);        // set the filter options
    	filter.setInputFormat(allData);       // prepare the filter for the data format
    	filter.setInvertSelection(false);  // do not invert the selection

    	// apply filter for test data here
    	Instances test = Filter.useFilter(allData, filter);

    	//  prepare and apply filter for training data here
    	filter.setInvertSelection(true);     // invert the selection to get other data
    	Instances train = Filter.useFilter(allData, filter);
    	Instances[] res = new Instances[] {train, test};
    	return res;
    }
}
