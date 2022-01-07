package com.predictionmarketing.RecommenderApp;

import java.io.File;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class EvaluateRecommender {
	public static void main(String[] args) throws Exception {
		DataModel model = new FileDataModel(new File("data/movies.csv"));
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder builder = new MyRecommenderBuilder();
		double result = evaluator.evaluate(builder, null, model, 0.7, 0.3);// train set , total data used
		System.out.println(result);
		 
		
	}

}


class MyRecommenderBuilder implements RecommenderBuilder{
	
	public Recommender buildRecommender (DataModel dataModel) throws TasteException{	
		
	 UserSimilarity similarity = new TanimotoCoefficientSimilarity(dataModel);
	 //UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
     UserNeighborhood neighborhood = new NearestNUserNeighborhood (10, similarity, dataModel);
     
     return new GenericUserBasedRecommender(dataModel, neighborhood,similarity);
}
}