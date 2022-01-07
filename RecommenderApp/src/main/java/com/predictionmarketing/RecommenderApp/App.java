package com.predictionmarketing.RecommenderApp;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

 
public class App 
{
    public static void main( String[] args ) throws IOException, TasteException
    {
        DataModel model = new FileDataModel(new File("data/movies.csv"));
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        UserSimilarity similarityBool = new LogLikelihoodSimilarity(model);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
        UserNeighborhood neighborhoodBool = new NearestNUserNeighborhood(10, similarityBool, model);
        UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood,similarity);
        UserBasedRecommender recommenderBool = new GenericBooleanPrefUserBasedRecommender(model, neighborhoodBool,similarityBool);
        List <RecommendedItem> recommendations = recommender.recommend(2, 10);
        List <RecommendedItem> recommendationsBool = recommenderBool.recommend(2, 10);
        System.out.println("For Generic User Based Recommender:");
        for (RecommendedItem recommendation : recommendations) {
        	System.out.println(recommendation); 	
        }
        System.out.println("For Generic Boolean Pref User Based Recommender:");
        for (RecommendedItem recommendation : recommendationsBool) {
        	System.out.println(recommendation);
        }
  
        
        
        
    }
}
