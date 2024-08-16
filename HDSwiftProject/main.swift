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
    
    let model = HDModel.buildHDModel(trainData: trainData, trainLabels: trainLabels, testData: testData, testLabels: testLabels, D: D, nLevels: nLevels, datasetName: "isolet")

    let config = MLModelConfiguration()
    config.computeUnits = .all // Or choose .cpuAndGPU if you're on a Mac
    
    guard let elementWiseAdder = try? ElementwiseAdd(configuration: config) else {
        fatalError("Failed to load model")
    }
    
    print(model.performElementWiseAddition(model: elementWiseAdder, inputArray1: [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0], inputArray2: [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]) ?? 6789998212)
    
//    let accuracy = model.trainNTimes(classHVs: model.classHVs, trainHVs: model.trainHVs, trainLabels: model.trainLabels, testHVs: model.testHVs, testLabels: model.testLabels, n: n)
    
//    print("the maximum accuracy is: " + String(accuracy.max()!) )
} else {
    print("Failed to load the data.")
}



