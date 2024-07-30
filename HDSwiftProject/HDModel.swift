import UIKit
import CoreML
import CreateMLComponents
import Foundation
//import CreateML

class HDModel {
    var trainData: [[Double]]
    var trainLabels: [Int]
    var testData: [[Double]]
    var testLabels: [Int]
    var D: Int
    var totalLevel: Int
    var posIdNum: Int
    var levelList: [Double]
    var levelHVs: [Int: [Int]]
    var IDHVs: [Int: [Int]]
    var trainHVs: Any // [[Int]]
    var testHVs: [[Int]]
    var classHVs: [[Int]]
    
    init(trainData: [[Double]], trainLabels: [Int], testData: [[Double]], testLabels: [Int], D: Int, totalLevel: Int) {
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
    
    public func getLevelList(trainData: [[Double]], totalLevel: Int) -> [Double] { // shud be double
        var minimum = trainData[0][0]
        var maximum = trainData[0][0]
        var levelList: [Double] = []
        
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
        let gap = length / Double(totalLevel)
        
        for lv in 0..<totalLevel {
            let value = Double(minimum) + Double(lv) * Double(gap)
            levelList.append(value)
        }
        levelList.append(Double(maximum))
        return levelList
    }
    
    func genLevelHVs(totalLevel: Int, D: Int) -> [Int: [Int]] {
        print("generating level HVs")
        var levelHVs: [Int: [Int]] = [:]
        let change = D / 2
        let nextLevel = D / 2 / totalLevel
        
        for level in 0..<totalLevel {
            var base = Array(repeating: -1, count: D)
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
    
    func genIDHVs(totalPos: Int, D: Int) -> [Int: [Int]] {
        print("Generating ID HVs")
        var IDHVs: [Int: [Int]] = [:]
        let change = D / 2
        
        guard change <= D else {
            fatalError("Are you seriooousss right neow bro")
        }
        
        for level in 0..<totalPos {
            var base = Array(repeating: -1, count: D)
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
    
//    func buildBufferHVs(mode: String, D: Int, dataset: String){
//        
//        if mode == "train"{
//            self.trainHVs = binarize(self.trainHVs, threshold: 2)
//            // convert trainHVs into array of doubles
//            self.trainHVs = convertToArrayOfArrayOfDoubles(from: trainHVs)
//            self.classHVs = oneHVPerClass(inputLabels: self.trainLabels, inputHVs: self.trainHVs, D: self.D)
//        }
//        else{
//            
//        }
//    }
    
    // Function to convert `Any` to `[[Double]]`
    func convertToArrayOfArrayOfDoubles(from value: Any) -> [[Double]]? {
        // Check if the value is of type `[[Double]]`
        if let doubleArray2D = value as? [[Double]] {
            return doubleArray2D
        }
        // Handle cases where the value is a nested array of strings
        else if let stringArray2D = value as? [[String]] {
            // Convert each inner array from `[String]` to `[Double]`
            let doubleArray2D = stringArray2D.map { row in
                row.compactMap { Double($0) }
            }
            return doubleArray2D
        }
        // Handle cases where the value is a nested array of integers
        else if let intArray2D = value as? [[Int]] {
            // Convert each inner array from `[Int]` to `[Double]`
            let doubleArray2D = intArray2D.map { row in
                row.map { Double($0) }
            }
            return doubleArray2D
        }
        // Handle cases where the value is a flat array of strings or doubles
        else if let flatArray = value as? [String] {
            let doubleArray = flatArray.compactMap { Double($0) }
            return [doubleArray]
        } else if let flatArray = value as? [Double] {
            return [flatArray]
        }
        // Handle cases where the value is a flat array of integers
        else if let flatArray = value as? [Int] {
            let doubleArray = flatArray.map { Double($0) }
            return [doubleArray]
        }
        // Handle other types as needed
        else {
            print("Error: The value is not of a supported type.")
            return nil
        }
    }
    
    
    // Function to convert `Any` to `[[Int]]`
    func convertToArrayOfIntArrays(from value: Any) -> [[Int]]? {
        // Check if the value is of type `[[Int]]`
        if let intArray2D = value as? [[Int]] {
            return intArray2D
        } else {
            // Handle the case where the type does not match
            print("not correct datatype convertToArrayOfIntArrays")
            return nil
        }
    }

    func binarize<T: Comparable & Numeric>(_ hypervector: Any, threshold: T) -> Any {

        func binarizeLevel(_ array: [Any], threshold: T) -> [Any] {
            return array.map { element in
                if let subArray = element as? [Any] {
                    return binarizeLevel(subArray, threshold: threshold)
                } else if let value = element as? T {
                    return value > threshold ? 1 : -1
                } else {
                    return element
                }
            }
        }
        
        if let hypervector = hypervector as? [Any] {
            return binarizeLevel(hypervector, threshold: threshold)
        } else {
            print("not an array")
            return hypervector
        }
    }
    
    
    func oneHVPerClass(inputLabels: [Int], inputHVs: [[Double]], D: Int) -> MLMultiArray? {
        let numClasses = (inputLabels.max() ?? 0) + 1
        
        // creates 2D multiarray [numClasses, D]
        guard let classHVs = try? MLMultiArray(shape: [NSNumber(value: numClasses), NSNumber(value: D)], dataType: .double) else {
            print("oneHVPerClass error")
            return nil
        }
        

        for i in 0..<numClasses {
            for j in 0..<D {
                classHVs[[i, j] as [NSNumber]] = NSNumber(value: 0.0)
            }
        }
        
        
        for (index, label) in inputLabels.enumerated() {
            let hv = inputHVs[index]
            for j in 0..<D {
                let currentValue = classHVs[[label, j] as [NSNumber]].doubleValue
                classHVs[[label, j] as [NSNumber]] = NSNumber(value: currentValue + hv[j]) // coreml operation
            }
        }
        
        return classHVs
    }
    

    
    
    func IDMultHV(inputBuffer: [Double], D: Int, inputHVs: [Int: [Double]], levelList: [Double], IDHVs: [Int:[Double]]) -> MLMultiArray?{
        
        
        var totalLevel = levelList.count - 1
        var totalPos = IDHVs.keys.count
        var sumHV = zeros(size: D)
        
        
        
        
        
        for keyVal in 0..<inputBuffer.count{
            let IDHV = IDHVs[keyVal]
            let key = numToKey(value: inputBuffer[keyVal], levelList: levelList)
            let levelHV = levelHVs[key]
            
            let dimensionIDHV = [NSNumber(value: IDHV!.count)]  // Shape should match the array size
            let dimensionlevelHV = [NSNumber(value: levelHV!.count)]
            if let IDHV = convertToMLMultiArrayDouble(doubleArray: IDHV!, dimension: dimensionIDHV), let levelHV = convertToMLMultiArray(intArray: levelHV!, dimension: dimensionlevelHV) {
                
                
//              sumHV = sumHV + (IDHV * levelHV)
                
                for i in 0..<IDHV.count {
                    let IDHV_val = IDHV[i].floatValue
                    let levelHV_val = levelHV[i].floatValue
                    sumHV![i] = IDHV_val * levelHV_val as NSNumber
                }
                
                
            } else {
                print("IDHV not converted")
            }
            
        }
        
        return sumHV
    }
    
    //
    func numToKey(value: Double, levelList: [Double]) -> Int{
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
    
    func zeros(size: Int) -> MLMultiArray? {
        do {
            let multiArray = try MLMultiArray(shape: [NSNumber(value: size)], dataType: .float32)
            
            // fill with zeros
            for index in 0..<multiArray.count {
                multiArray[index] = 0
            }
            
            return multiArray
        } catch {
            print("couldn't create ml array: \(error)")
            return nil
        }
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
    
    func convertToMLMultiArrayDouble(doubleArray: [Double], dimension: [NSNumber]) -> MLMultiArray? {
        do {
            let multiArray = try MLMultiArray(shape: dimension, dataType: .float32)
            for (index, value) in doubleArray.enumerated() {
                multiArray[index] = NSNumber(value: Float(value))
            }
            return multiArray
        } catch {
            print("couldn't create ml array: \(error)")
            return nil
        }
    }

    
}
