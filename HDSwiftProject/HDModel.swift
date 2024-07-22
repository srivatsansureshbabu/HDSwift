import UIKit
import CoreML
import CreateMLComponents
import Foundation

class HDModel {
    var trainData: [[Double]]
    var trainLabels: [Int]
    var testData: [[Double]]
    var testLabels: [Int]
    var D: Int
    var totalLevel: Int
    var posIdNum: Double
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
        self.posIdNum = Double(trainData.count)
        self.trainHVs = []
        self.testHVs = []
        self.classHVs = []
        self.levelList = []
        self.levelHVs = [:]
        self.IDHVs = [:]
        self.levelList = getLevelList(trainData: trainData, totalLevel: totalLevel)
        self.levelHVs = genLevelHVs(totalLevel: totalLevel, D: D)
        self.IDHVs = genIDHVs(totalLevel: Int(posIdNum), D: D)
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
        let gap = Int(length) / totalLevel
        
        for lv in 0..<totalLevel {
            let value = Double(minimum) + Double(lv * gap)
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
    
    func genIDHVs(totalLevel: Int, D: Int) -> [Int: [Int]] {
        print("Generating ID HVs")
        var IDHVs: [Int: [Int]] = [:]
        let change = D / 2
        
        // Ensure `change` is less than `D`
        guard change <= D else {
            fatalError("The value of `D` should be at least twice the value of `change`.")
        }
        
        for level in 0..<totalLevel {
            var base = Array(repeating: -1, count: D)
            var toOne: [Int] = []
            
            // Populate `toOne` with random indices
            for _ in 1..<D {
                let randomNumber = Int.random(in: 0..<D)
                toOne.append(randomNumber)
            }
            
            // Shuffle and take the first `change` indices
            toOne.shuffle()
            toOne = Array(toOne.prefix(change))
            
            // Set the positions in `base` to 1
            for index in toOne {
                base[index] = 1
            }
            
            // Store the result in `IDHVs`
            IDHVs[level] = base
        }
        
        return IDHVs
    }

//    func genIDHVs(totalPos: Double, D: Int) -> [Int: [Int]] {
//        print("generating ID HVs")
//        var IDHVs: [Int: [Int]] = [:]
//        let change = D / 2
//        
//        for level in 0..<totalLevel {
//            var base = Array(repeating: -1, count: D)
//            var toOne: [Int] = []
//            
//            for _ in 1..<D {
//                let randomNumber = Int.random(in: 0..<D)
//                toOne.append(randomNumber)
//            }
//            
//            
//            
//            toOne.shuffle()
//            toOne = Array(toOne.prefix(change))
//            
//            for index in toOne {
//                base[index] = 1
//            }
//            IDHVs[level] = base
//        }
//        return IDHVs
//    }
}
