import UIKit
import CoreML
import CreateMLComponents

class HDModel {
    var trainData: [[Double]]
    var trainLabels: [Int]
    var testData: [[Double]]
    var testLabels: [Int]
    var D: Int
    var totalLevel: Int
    var posIdNum: Double
    var levelList: [Int] = [0]
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
        self.posIdNum = trainData[0][0]
        self.trainHVs = []
        self.testHVs = []
        self.classHVs = []
        self.levelList = getLevelList(trainData: trainData, totalLevel: totalLevel)
        self.levelHVs = genLevelHVs(totalLevel: totalLevel, D: D)
        self.IDHVs = genIDHVs(totalPos: NSNumber(value: posIdNum), D: D)
    }
    
    func getLevelList(trainData: [[Double]], totalLevel: Int) -> [Int] {
        var minimum = trainData[0][0]
        var maximum = trainData[0][0]
        var levelList: [Int] = []
        
        for item in trainData {
            let localMin = item.min() ?? 0
            let localMax = item.max() ?? 0
            
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
            let value = Int(minimum) + lv * gap
            levelList.append(value)
        }
        levelList.append(Int(maximum))
        return levelList
    }
    
    func genLevelHVs(totalLevel: Int, D: Int) -> [Int: [Int]] {
        print("Generating level HVs")
        var levelHVs: [Int: [Int]] = [:]
        let nextLevel = Int(D / 2 / totalLevel)
        let change = Int(D / 2)
        
        for level in 0..<totalLevel {
            var base = Array(repeating: -1, count: D)
            var toOne: [Int] = []
            
            for _ in 0..<D {
                let randomNumber = Int.random(in: 1...D)
                toOne.append(randomNumber)
            }
            
            toOne.shuffle()
            let countToChange = (level == 0) ? change : nextLevel
            for index in 0..<countToChange {
                base[toOne[index]] *= -1
            }
            
            levelHVs[level] = base
        }
        
        return levelHVs
    }
    
    func genIDHVs(totalPos: NSNumber, D: Int) -> [Int: [Int]] {
        print("Generating ID HVs")
        var IDHVs: [Int: [Int]] = [:]
        let change = Int(D / 2)
        
        for level in 0..<totalLevel {
            var base = Array(repeating: -1, count: D)
            var toOne: [Int] = []
            
            for _ in 0..<D {
                let randomNumber = Int.random(in: 1...D)
                toOne.append(randomNumber)
            }
            
            toOne.shuffle()
            for index in 0..<change {
                base[toOne[index]] = 1
            }
            
            IDHVs[level] = base
        }
        
        return IDHVs
    }
}
