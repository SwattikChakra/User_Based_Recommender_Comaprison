package com.predictionmarketing.RecommenderApp;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class GenericBooleanPref {

	public static void main(String[] args) throws TasteException, IOException {
		DataModel model = new FileDataModel(new File("data/movies.csv"));
		RecommenderIRStatsEvaluator evaluator =
				 new GenericRecommenderIRStatsEvaluator();
				RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
				 @Override
				 public Recommender buildRecommender(DataModel model) throws TasteException {
				 //UserSimilarity similarity = new TanimotoCoefficientSimilarity(model);
				 UserSimilarity similarity = new LogLikelihoodSimilarity(model);
				 UserNeighborhood neighborhood =
				 new NearestNUserNeighborhood(10, similarity, model);
				 return new  GenericBooleanPrefUserBasedRecommender(
				 model, neighborhood, similarity);
				 }
				};
								
					IRStatistics stats = evaluator.evaluate(
							 recommenderBuilder, null, model, null, 10,
							 GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD,
							 0.3);
					System.out.println(stats.getPrecision());
					System.out.println(stats.getRecall());
			

}
}
