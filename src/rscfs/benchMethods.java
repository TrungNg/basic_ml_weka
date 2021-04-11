/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rscfs;
import weka.core.Instances;
import weka.attributeSelection.*;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
//import weka.attributeSelection.PSOSearch;


/**
 *
 * @author truongtran
 */
public class benchMethods 
{
    //Remove features not in list
    private static Instances RemoveFS(Instances inst, int[] list)
    {
        Instances newInst=new Instances(inst);
        boolean[] check=new boolean[inst.numAttributes()];
        for(int i=0;i<list.length;i++)
            check[list[i]]=true;
        for(int i=inst.numAttributes()-1;i>=0;i--)
            if(!check[i])
                newInst.deleteAttributeAt(i);
        return newInst;
        
    }
    //Using Best First Search
    public static Instances[] bfsCfs(Instances trainInsts, Instances testInsts) throws Exception
    {
        AttributeSelection attSelect = new AttributeSelection();
        BestFirst bfs = new BestFirst();
        attSelect.setSearch(bfs);
        CfsSubsetEval cfs = new CfsSubsetEval();
        attSelect.setEvaluator(cfs);
        attSelect.SelectAttributes(trainInsts);
        int[] list = attSelect.selectedAttributes();
        Instances[] results = new Instances[2];
        results[0] = RemoveFS(trainInsts, list);
        results[1] = RemoveFS(testInsts, list);
        return results;
        
    }
    
}
