//
//  main.swift
//  HDSwiftProject
//
//  Created by Hanna Silva on 8/15/24.
//

import Foundation
import CoreML

func loadIsoletData() -> ([[Float]], [Int], [[Float]], [Int])? {
    // Access the file in the app bundle
    if let filePath = Bundle.main.path(forResource: "isolet_train", ofType: "json") {
        print("File path: \(filePath)")
        
        do {
            // Read the file data
            let fileData = try Data(contentsOf: URL(fileURLWithPath: filePath))
            
            // Decode the JSON data
            let decoder = JSONDecoder()
            let isoletData = try decoder.decode(IsoletData.self, from: fileData)
            
            // Extract features and labels for trainData
            let trainFeaturesDict = isoletData.trainData.features
            let trainLabelsArray = isoletData.trainData.labels
            
            // Convert dictionary to array for trainData
            var trainData: [[Float]] = []
            let sortedTrainKeys = trainFeaturesDict.keys.sorted { Int($0)! < Int($1)! }
            
            for key in sortedTrainKeys {
                if let features = trainFeaturesDict[key] {
                    trainData.append(features)
                }
            }
            
            // Extract trainLabels directly
            let trainLabels = trainLabelsArray
            
            // Extract features and labels for testData
            let testFeaturesDict = isoletData.testData.features
            let testLabelsArray = isoletData.testData.labels
            
            // Convert dictionary to array for testData
            var testData: [[Float]] = []
            let sortedTestKeys = testFeaturesDict.keys.sorted { Int($0)! < Int($1)! }
            
            for key in sortedTestKeys {
                if let features = testFeaturesDict[key] {
                    testData.append(features)
                }
            }
            
            // Extract testLabels directly
            let testLabels = testLabelsArray
            
            // Return the data
            return (trainData, trainLabels, testData, testLabels)
            
        } catch {
            print("Failed to load or decode JSON with error: \(error)")
            return nil
        }
    } else {
        print("File not found in bundle")
        return nil
    }
}


let D = 10
let nLevels = 100
let n = 10

if let (trainData, trainLabels, testData, testLabels) = loadIsoletData() {
    
    let config = MLModelConfiguration()
    config.computeUnits = .all
    
    let elementWiseMultiplier = try ElementWiseMultiplication617x10(configuration: config)
//    let elementWiseMultiplier = try MultiplyAndSumModel(configuration: config)
    let elementWiseAdder = try ElementWiseAddition_2D_to_1D_617x10(configuration: config)
//
    let model = HDModel.buildHDModel(modelMultiply: elementWiseMultiplier, modelAdd: elementWiseAdder, trainData: trainData, trainLabels: trainLabels, testData: testData, testLabels: testLabels, D: D, nLevels: nLevels, datasetName: "isolet")

//    let model = HDModel(trainData: trainData, trainLabels: trainLabels, testData: testData, testLabels: testLabels, D: D, totalLevel: nLevels)
    
    
    // Test Case
//    let inputBuffer: [Float] = [1, 2]
//    let D = 3
//    let levelHVs: [Int: [Float]] = [
//        0: [1.0, 0.0, 1.0],
//        1: [0.0, 1.0, 0.0],
//        2: [1.0, 1.0, 1.0]
//    ]
//    let levelList: [Float] = [0, 1, 2]
//    let IDHVs: [Int: [Float]] = [
//        0: [1.0, 1.0, 1.0],
//        1: [0.0, 1.0, 0.0]
//    ]
     
    // Running the function
//    let result = model.IDMultHVNew(modelMultiply: elementWiseMultiplier, modelAdd: elementWiseAdder, inputBuffer: inputBuffer, D: D, levelHVs: levelHVs, levelList: levelList, IDHVs: IDHVs)
//    print(result!)   Expected Output: [1.0, 1.0, 1.0]
//    print(model.performElementWiseMultiplication(model: elementWiseMultiplier, inputArray1: [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], inputArray2: [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]) ?? 6789998212)
//
//    
//
    let accuracy = model.trainNTimes(classHVs: model.classHVs, trainHVs: model.trainHVs, trainLabels: model.trainLabels, testHVs: model.testHVs, testLabels: model.testLabels, n: n)
    
    print("the maximum accuracy is: " + String(accuracy.max()!) )
} else {
    print("Failed to load the data.")
}



