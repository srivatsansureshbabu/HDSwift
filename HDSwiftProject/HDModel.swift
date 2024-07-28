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
    var trainHVs: [[Int]]
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
//            self.trainHVs = binarize(
//        }
//        else{
//            
//        }
//    }
        

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
            print("boy wut da hael boy")
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
    

    
    
    func IDMultHV(inputBuffer: [[Double]], D: Int, inputHVs: [Int: [Int]], levelList: [Double], IDHVs: [Int:[Int]]) -> MLMultiArray?{
        
        
        var totalLevel = levelList.count - 1
        var totalPos = IDHVs.keys.count
        var sumHV = zeros(size: D)
        
        
        for keyVal in 0...inputBuffer.count{
            var IDHV = IDHVs[keyVal]
//            var key = numToKey(inputBuffer[keyVal], levelList)
//            var levelHV = levelHVs[key]
//            sumHV = sumHV + (IDHV * levelHV) // may be a coreml operation
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
        var keyIndex = 0
        
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
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }
    
    
    
}
