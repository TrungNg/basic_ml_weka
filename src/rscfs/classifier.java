/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rscfs;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author truongtran
 */
public class classifier
{
    public static double J48(int rseed, Instances trainInsts, Instances testInsts) throws Exception
    {
        J48 classifier=new J48();
    	classifier.setSeed(rseed);
        classifier.buildClassifier(trainInsts);
        Evaluation eval=new Evaluation(trainInsts);
        eval.evaluateModel(classifier, testInsts);
        System.out.println(eval.toSummaryString("\nResult using J48:\n",true));
		return eval.pctCorrect();
    }

    public static double J48CV(int rseed, Instances allData) throws Exception
    {
    	J48 classifier = new J48();
    	classifier.setSeed(rseed);
		Evaluation eval = new Evaluation(allData);
		eval.crossValidateModel(classifier, allData, 10, new Random(0));
		System.out.println(eval.toSummaryString("\nResult using C4.5 (J48) with 10-fold CV:\n",true));
		return eval.pctCorrect();
    }

    public static double NB(Instances trainInsts, Instances testInsts) throws Exception
    {
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(trainInsts);
        Evaluation eval=new Evaluation(trainInsts);
        eval.evaluateModel(classifier, testInsts);
        System.out.println(eval.toSummaryString("\nResult using NB:\n",true));
		return eval.pctCorrect();
    }

    public static double NBCV(Instances allData) throws Exception
    {
        NaiveBayes classifier=new NaiveBayes();
		Evaluation eval = new Evaluation(allData);
		eval.crossValidateModel(classifier, allData, 10, new Random(0));
		System.out.println(eval.toSummaryString("\nResult using Naive Bayes with 10-fold CV:\n",true));
		return eval.pctCorrect();
    }

	public static double MLPCV(int rseed, Instances allData) throws Exception
	{
		MultilayerPerceptron classifier = new MultilayerPerceptron();
		//classifier.setHiddenLayers("10,6");
    	//classifier.setLearningRate(0.3);
    	classifier.setSeed(rseed);
    	//classifier.setTrainingTime(500);
//    	String[] options = classifier.getOptions();
//    	for (String s: options) {
//            System.out.print(s);
//        }
		Evaluation eval = new Evaluation(allData);
		eval.crossValidateModel(classifier, allData, 10, new Random(0));
		System.out.println(eval.toSummaryString("\nResult using MLP with 10-fold CV:\n",true));
		return eval.pctCorrect();
	}
    public static double MLP(int rseed, Instances trainInsts, Instances testInsts) throws Exception
    {
    	MultilayerPerceptron classifier=new MultilayerPerceptron();
    	//classifier.setHiddenLayers("10,6");
    	//System.out.println(classifier.getHiddenLayers());
    	//classifier.setLearningRate(0.3);
    	classifier.setSeed(rseed);
    	//classifier.setTrainingTime(500);
        classifier.buildClassifier(trainInsts);
        Evaluation eval=new Evaluation(trainInsts);
        eval.evaluateModel(classifier, testInsts);
        System.out.println(eval.toSummaryString("\nResult using MLP:\n",true));
        return eval.pctCorrect();
    }

    public static double kNN(Instances trainInsts, Instances testInsts) throws Exception
    {
        IBk classifier=new IBk();
        classifier.buildClassifier(trainInsts);
        Evaluation eval=new Evaluation(trainInsts);
        eval.evaluateModel(classifier, testInsts);
        System.out.println(eval.toSummaryString("\nResult using KNN:\n",true));
        return eval.pctCorrect();
    }

    public static double kNNCV(Instances allData) throws Exception
    {
    	IBk classifier=new IBk();
		Evaluation eval = new Evaluation(allData);
		eval.crossValidateModel(classifier, allData, 10, new Random(0));
		System.out.println(eval.toSummaryString("\nResult using kNN (IBk) with 10-fold CV:\n",true));
		return eval.pctCorrect();
    }

    public static double SMO(Instances trainInsts, Instances testInsts) throws Exception
    {
        SMO classifier=new SMO();
        classifier.buildClassifier(trainInsts);
        Evaluation eval=new Evaluation(trainInsts);
        eval.evaluateModel(classifier, testInsts);
        System.out.println(eval.toSummaryString("\nResult using SVM:\n",true));
        return eval.pctCorrect();
    }

    public static double SMOCV(Instances allData) throws Exception
    {
    	SMO classifier=new SMO();
		Evaluation eval = new Evaluation(allData);
		eval.crossValidateModel(classifier, allData, 10, new Random(0));
		System.out.println(eval.toSummaryString("\nResult using SVM (SM0) with 10-fold CV:\n",true));
		return eval.pctCorrect();
    }

    public static double RandomForest(int rseed, Instances trainInsts, Instances testInsts, int newNumTrees) throws Exception
    {
        RandomForest classifier=new RandomForest();
        classifier.setNumIterations(newNumTrees);
        classifier.buildClassifier(trainInsts);
        Evaluation eval=new Evaluation(trainInsts);
        eval.evaluateModel(classifier, testInsts);
        System.out.println(eval.toSummaryString("\nResult using Random Forest:\n",true));
        return eval.pctCorrect();
    }

    public static double RandomForestCV(int rseed, Instances allData, int newNumTrees) throws Exception
    {
    	RandomForest classifier=new RandomForest();
    	classifier.setSeed(rseed);
        classifier.setNumIterations(newNumTrees);
		Evaluation eval = new Evaluation(allData);
		eval.crossValidateModel(classifier, allData, 10, new Random(0));
		System.out.println(eval.toSummaryString("\nResult using RandomForest with 10-fold CV:\n",true));
		return eval.pctCorrect();
    }
}
