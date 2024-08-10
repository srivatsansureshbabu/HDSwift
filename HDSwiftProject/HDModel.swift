import UIKit
import CoreML
import CreateMLComponents
import Foundation
//import CreateML

public class HDModel {
    var trainData: [[Float]]
    var trainLabels: [Int]
    var testData: [[Float]]
    var testLabels: [Int]
    var D: Int
    var totalLevel: Int
    var posIdNum: Int
    var levelList: [Float]
    var levelHVs: [Int: [Float]]
    var IDHVs: [Int: [Float]]
    var trainHVs: [[Float]]
    var testHVs: [[Float]]
    var classHVs: [[Float]]
    
    init(trainData: [[Float]], trainLabels: [Int], testData: [[Float]], testLabels: [Int], D: Int, totalLevel: Int) {
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.testData = testData
        self.testLabels = testLabels
        self.D = D
        self.totalLevel = totalLevel
        self.posIdNum = (trainData[0].count)
        self.trainHVs = []
        self.testHVs = []
        self.classHVs = []
        self.levelList = []
        self.levelHVs = [:]
        self.IDHVs = [:]
        self.levelList = getLevelList(trainData: trainData, totalLevel: totalLevel)
        self.levelHVs = genLevelHVs(totalLevel: totalLevel, D: D)
        self.IDHVs = genIDHVs(totalPos: posIdNum, D: D)
    }
    
    
    
    
    
    
    func buildBufferHVs(mode: String, D: Int, dataset: String){
        
        if mode == "train"{
            
            print("Encoding Training Data")
            for i in 0..<self.trainData.count{
                self.trainHVs.append(IDMultHV(inputBuffer: self.trainData[i], D: D, levelHVs: self.levelHVs, levelList: self.levelList, IDHVs: self.IDHVs)!)
            }
                
                
                
            self.trainHVs = binarize(array: self.trainHVs)
            // convert trainHVs into array of doubles
            self.trainHVs = convertToArrayOfArrayOfFloats(from: trainHVs)!
            let IntIntclassHVs = oneHVPerClass(inputLabels: self.trainLabels, inputHVs: self.trainHVs, D: self.D)
            self.classHVs = convertToArrayOfArrayOfFloats(from: IntIntclassHVs!)!
            
            
        }
        else{
            print("Encoding Testing Data")
            
            for index in 0..<testData.count{
                self.testHVs.append(IDMultHV(inputBuffer: self.testData[index], D: self.D, levelHVs: self.levelHVs, levelList: self.levelList, IDHVs: self.IDHVs)!)
            }
            let DoubleDoubleTestHVs = binarize(array: self.testHVs)
            self.testHVs = convertToArrayOfArrayOfFloats(from: DoubleDoubleTestHVs)!
        }
    }
//    func buildBufferHVs(mode: String, D: Int, dataset: String) {
//          let fileManager = FileManager.default
//          let directoryPath = "./../dataset/\(dataset)/"
//          let fileName = "\(mode)_bufferHVs_\(D).json"
//          let filePath = directoryPath + fileName
//          if mode == "train" {
//              if fileManager.fileExists(atPath: filePath) {
//                  print("Loading Encoded Training Data")
//                  if let data = fileManager.contents(atPath: filePath) {
//                      do {
//                          self.trainHVs = try JSONDecoder().decode([[Float]].self, from: data)
//                      } catch {
//                          print("Error decoding training data: \(error)")
//                      }
//                  }
//              } else {
//                  print("Encoding Training Data")
//                  for item in trainData {
//                      self.trainHVs.append(IDMultHV(inputBuffer: item, D: D, levelHVs: levelHVs, levelList: levelList, IDHVs: IDHVs)!)
//                  }
//                  if let data = try? JSONEncoder().encode(self.trainHVs) {
//                      fileManager.createFile(atPath: filePath, contents: data, attributes: nil)
//                  }
//              }
//              self.trainHVs = binarize(array: self.trainHVs)
//              self.classHVs = convertToArrayOfArrayOfFloats(from: oneHVPerClass(inputLabels: self.trainLabels, inputHVs: self.trainHVs, D: D)!)!
//              
//          } else {
//              let fileName = "test_bufferHVs_\(D).json"
//              let filePath = directoryPath + fileName
//              if fileManager.fileExists(atPath: filePath) {
//                  print("Loading Encoded Testing Data")
//                  if let data = fileManager.contents(atPath: filePath) {
//                      do {
//                          self.testHVs = try JSONDecoder().decode([[Float]].self, from: data)
//                      } catch {
//                          print("Error decoding testing data: \(error)")
//                      }
//                  }
//              } else {
//                  print("Encoding Testing Data")
//                  for item in testData {
//                      self.testHVs.append(IDMultHV(inputBuffer: item, D: D, levelHVs: levelHVs, levelList: levelList, IDHVs: IDHVs)!)
//                  }
//                  if let data = try? JSONEncoder().encode(self.testHVs) {
//                      fileManager.createFile(atPath: filePath, contents: data, attributes: nil)
//                  }
//              }
//              self.testHVs = binarize(array: self.testHVs)
//          }
//      }
    
    
    func getLevelList(trainData: [[Float]], totalLevel: Int) -> [Float] { // shud be double
        var minimum = trainData[0][0]
        var maximum = trainData[0][0]
        var levelList: [Float] = []
        
        for item in trainData {
            let localMin = item.min() ?? minimum
            let localMax = item.max() ?? maximum
            
            if localMin < minimum {
                minimum = localMin
            }
            if localMax > maximum {
                maximum = localMax
            }
        }
        
        let length = maximum - minimum
        let gap = length / Float(totalLevel)
        
        for lv in 0..<totalLevel {
            let value = Float(minimum) + Float(lv) * Float(gap)
            levelList.append(value)
        }
        levelList.append(Float(maximum))
        return levelList
    }
    
    
    func genLevelHVs(totalLevel: Int, D: Int) -> [Int: [Float]] {
        print("generating level HVs")
        var levelHVs: [Int: [Float]] = [:]
        let change = D / 2
        let nextLevel = D / 2 / totalLevel
        
        for level in 0..<totalLevel {
            let baseInt = Array(repeating: -1, count: D)
            var base = baseInt.map { Float($0) }

            var toOne: [Int] = []
            
            if level == 0 {
                for _ in 1..<D {
                    let randomNumber = Int.random(in: 0..<D)
                    toOne.append(randomNumber)
                }
                toOne.shuffle()
                toOne = Array(toOne.prefix(change))
            } else {
                for _ in 1..<D {
                    let randomNumber = Int.random(in: 0..<D)
                    toOne.append(randomNumber)
                }
                toOne.shuffle()
                toOne = Array(toOne.prefix(nextLevel))
            }
            
            for index in toOne {
                base[index] *= -1
            }
            levelHVs[level] = base
        }
        return levelHVs
    }
    
    func genIDHVs(totalPos: Int, D: Int) -> [Int: [Float]] {
        print("Generating ID HVs")
        var IDHVs: [Int: [Float]] = [:]
        let change = D / 2
        
        guard change <= D else {
            fatalError("genIDHVs error")
        }
        
        for level in 0..<totalPos {
            let baseInt = Array(repeating: -1, count: D)
            var base = baseInt.map { Float($0) }
            var toOne: [Int] = []
            
            for _ in 1..<D {
                let randomNumber = Int.random(in: 0..<D)
                toOne.append(randomNumber)
            }
            
            toOne.shuffle()
            toOne = Array(toOne.prefix(change))
            
            for index in toOne {
                base[index] = 1
            }
            
            IDHVs[level] = base
        }
        
        return IDHVs
    }
    
    func binarize(array:[Float]) -> [Float]{
        var binarizedArray = [Int](repeating: 0, count: array.count)
        for i in 0..<array.count{
            if array[i] > 0{
                binarizedArray[i] = 1
            }
            else{
                binarizedArray[i] = -1
            }
        }
        
        return array
        
    }
    
    
    func binarize(array: [[Float]]) -> [[Float]]{
        
        let rows = array.count
        let columns = array.first?.count ?? 0

        var binarizedArray = zeros2D(columns: columns, rows: rows)
        
        for i in 0..<array.count{
            for j in 0..<array[i].count{
                if array[i][j] > 0{
                    binarizedArray[i][j] = 1.0
                }
                else{
                    binarizedArray[i][j] = -1.0
                }
            }
        }
        
        return binarizedArray
    }
    
    
    func oneHVPerClass(inputLabels: [Int], inputHVs: [[Float]], D: Int) -> MLMultiArray? {
        let numClasses = (inputLabels.max() ?? 0) + 1
        
        // creates 2D multiarray [numClasses, D]
        guard let classHVs = try? MLMultiArray(shape: [NSNumber(value: numClasses), NSNumber(value: D)], dataType: .float32) else {
            print("oneHVPerClass error")
            return nil
        }
        
        
        for (index, label) in inputLabels.enumerated() {
            let hv = inputHVs[index] // array
            for j in 0..<D {
                let currentValue = classHVs[[label, j] as [NSNumber]].floatValue
                classHVs[[label, j] as [NSNumber]] = NSNumber(value: currentValue + hv[j]) // coreml operation //
            }
        }
        
        return classHVs
    }
    

    
    
    func IDMultHV(inputBuffer: [Float], D: Int, levelHVs: [Int: [Float]], levelList: [Float], IDHVs: [Int:[Float]]) -> [Float]?{
        
        
//        var totalLevel = levelList.count - 1
//        var totalPos = IDHVs.keys.count
        var sumHV = zeros(size: D)!
        
        for keyVal in 0..<inputBuffer.count{
            let IDHV = IDHVs[keyVal]
            let key = numToKey(value: inputBuffer[keyVal], levelList: levelList)
            let levelHV = levelHVs[key]
            
            for i in 0..<IDHV!.count {
                sumHV[i] += IDHV![i] * levelHV![i]
            }
        }
        
        return sumHV
    }
    
    func numToKey(value: Float, levelList: [Float]) -> Int{
        let levelListValue = levelList.last
        
        if value == levelListValue {
            return levelList.count - 2
        }
        var upperIndex = levelList.count - 1
        var lowerIndex = 0
        let keyIndex = 0
        
        while upperIndex > lowerIndex{
            var keyIndex = Int((upperIndex+lowerIndex)/2)
            
            if levelList[keyIndex] <= value && levelList[keyIndex+1] > value{
                return keyIndex
            }
            if levelList[keyIndex] > value{
                upperIndex = keyIndex
                keyIndex = Int((upperIndex + lowerIndex)/2)
            }
            else{
               lowerIndex = keyIndex
                keyIndex = Int((upperIndex+lowerIndex)/2)
            }
        }
        return keyIndex
    }
    
    func hammingDistance(x: [Float], y: [Float]) -> Float{
        var count = 0.0
        for i in 0..<x.count{
            if( x[i] != y[i]){
                count = count + 1.0
            }
        }

        return Float(x.count) - Float(count)
    }
    func checkVector(classHVs: [[Float]], inputHV: [Float]) -> Int{
        var guess = -1
        var maximum = Int.min
        var count : [Int:Int] = [:]
        
        for key in 0..<classHVs.count{
            count[key] = Int(hammingDistance(x:classHVs[key], y:inputHV))
            if count[key]! > maximum{
                guess = key
                maximum = count[key]!
            }
        }
        return guess
    }
    
    // error rate doesn't work properly
    func trainOneTime(classHVs: [[Float]], trainHVs: [[Float]], trainLabels: [Int]) -> ([[Float]], Float){
        
        var retClassHVs = classHVs
        let copyClassHVs = classHVs
        var classHVs_binary = binarize(array: copyClassHVs)
        classHVs_binary = convertToArrayOfArrayOfFloats(from: classHVs_binary)!
        var wrong_num = 0
        
        for index in 0..<trainLabels.count{
            let guess = checkVector(classHVs: classHVs_binary, inputHV: trainHVs[index])
            
            if !(trainLabels[index] == guess){
                wrong_num = wrong_num + 1

                for i in 0..<retClassHVs[guess].count{
                    retClassHVs[guess][i] = retClassHVs[guess][i] - trainHVs[index][i]
                } // element wise subtraction
                
                for (i,x) in trainHVs[index].enumerated(){
                    retClassHVs[trainLabels[index]][i] = retClassHVs[trainLabels[index]][i] + x
                } // element wise addition
                
            }
        }
        let error = Float(wrong_num)/Float(trainLabels.count)
        print("Error: " + String(error))
        return (retClassHVs,error)
    }
   
    func trainNTimes(classHVs: [[Float]], trainHVs: [[Float]], trainLabels: [Int], testHVs: [[Float]], testLabels: [Int], n: Int) -> [Float]{

        var accuracy: [Float] = []
        var currClassHVs = classHVs
        accuracy.append(test(classHVs: currClassHVs,testHVs: testHVs,testLabels: testLabels))
        
        for i in 0..<n{
            print("iteration: " + String(i))
            let (currClassHVs,error) = trainOneTime(classHVs: currClassHVs, trainHVs: trainHVs, trainLabels: trainLabels)
            accuracy.append(test(classHVs: currClassHVs, testHVs: testHVs, testLabels: testLabels))
        }
        return accuracy
    }
    
    func test (classHVs: [[Float]], testHVs: [[Float]], testLabels: [Int]) -> Float{
        let classHVs_binary = binarize(array: classHVs)
        var correct = 0
        for index in 0..<testHVs.count{
            let guess = checkVector(classHVs: classHVs_binary, inputHV: testHVs[index])
            if testLabels[index] == guess {
                correct += 1
            }
        }
        let accuracy = (Float(correct) / Float(testLabels.count) )*100
        print("The accuracy is: " + String (accuracy))
        return Float((accuracy))
    }
    
    public static func buildHDModel(trainData: [[Float]], trainLabels: [Int], testData: [[Float]], testLabels: [Int], D: Int, nLevels: Int, datasetName: String) -> HDModel{
        let model = HDModel(trainData: trainData, trainLabels: trainLabels, testData: testData, testLabels: testLabels, D: D, totalLevel: nLevels)
        model.buildBufferHVs(mode: "train", D: D, dataset: datasetName)
        model.buildBufferHVs(mode: "test", D: D, dataset: datasetName)
        
        return model
    }
    
    
    
    
    
    
    
    
    
    func zeros(size: Int) -> [Float]? {
        // Check if size is non-negative
        guard size >= 0 else {
            print("Size must be non-negative.")
            return nil
        }
        
        // Create an array of doubles filled with zeros
        let zeroArray = [Float](repeating: 0.0, count: size)
        
        return zeroArray
    }
    
    func zeros2D(columns: Int, rows: Int) -> [[Float]] {
        
        let zerosArray: [[Float]] = Array(repeating: Array(repeating: 0, count: columns), count: rows)

        // Create an array of doubles filled with zeros
        
        return zerosArray
    }

    
    func convertToMLMultiArray(intArray: [Int], dimension: [NSNumber]) -> MLMultiArray? {
        do {
            // create an MLMultiArray with the specified shape and data type (e.g., float32)
            let multiArray = try MLMultiArray(shape: dimension, dataType: .float32)
            
            // populate MLMultiArray with values from the integer array
            for (index, value) in intArray.enumerated() {
                multiArray[index] = NSNumber(value: Float(value))
            }
            
            return multiArray
        } catch {
            print("couldn't create ml array: \(error)")
            return nil
        }
    }
    
    func convertToMLMultiArrayFloat(floatArray: [Float], dimension: [NSNumber]) -> MLMultiArray? {
        do {
            let multiArray = try MLMultiArray(shape: dimension, dataType: .float32)
            for (index, value) in floatArray.enumerated() {
                multiArray[index] = NSNumber(value: Float(value))
            }
            return multiArray
        } catch {
            print("couldn't create ml array: \(error)")
            return nil
        }
    }
    
    // Function to convert Any to [[Double]]
    func convertToArrayOfArrayOfFloats(from value: Any) -> [[Float]]? {
        // Check if the value is of type [[Double]]
        if let floatArray2D = value as? [[Float]] {
            return floatArray2D
        }
        // Handle cases where the value is a nested array of strings
        else if let stringArray2D = value as? [[String]] {
            // Convert each inner array from [String] to [Double]
            let floatArray2D = stringArray2D.map { row in
                row.compactMap { Float($0) }
            }
            return floatArray2D
        }
        // Handle cases where the value is a nested array of integers
        else if let intArray2D = value as? [[Int]] {
            // Convert each inner array from [Int] to [Double]
            let floatArray2D = intArray2D.map { row in
                row.map { Float($0) }
            }
            return floatArray2D
        }
        // Handle cases where the value is a flat array of strings or doubles
        else if let flatArray = value as? [String] {
            let floatArray = flatArray.compactMap { Float($0) }
            return [floatArray]
        } else if let flatArray = value as? [Float] {
            return [flatArray]
        }
        // Handle cases where the value is a flat array of integers
        else if let flatArray = value as? [Int] {
            let floatArray = flatArray.map { Float($0) }
            return [floatArray]
        }
        // Handle other types as needed
        else {
            print("Error: The value is not of a supported type.")
            return nil
        }
    }
    
    
    // Function to convert Any to [[Int]]
    func convertToArrayOfIntArrays(from value: Any) -> [[Int]]? {
        // Check if the value is of type [[Int]]
        if let intArray2D = value as? [[Int]] {
            return intArray2D
        } else {
            // Handle the case where the type does not match
            print("not correct datatype convertToArrayOfIntArrays")
            return nil
        }
    }

    
}
