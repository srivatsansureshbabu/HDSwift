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
    var levelHVs: [String: Int]
    var IDHVs: [Double]
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
        self.IDHVs = genIDHVs(totalPos: posIdNum, D: D)
    }
    
    func getLevelList(trainData: [[Double]], totalLevel: Int) -> [Int]{
        var minimum = trainData[0][0]
        var maximum = trainData[0][0]
        var levelList: [Int] = []
        
        for item in trainData {
            
            let localMin = trainData.flatMap({ $0 }).min()
            let localMax = trainData.flatMap({ $0 }).max()
            
            if localMin! < minimum {
                minimum = localMin!
            }
            if localMax! > maximum {
                maximum = localMax!
            }
        }
        
        var length = maximum - minimum
        var gap = Int(length) / totalLevel
        
        for lv in stride(from:0, to: totalLevel, by: 1) {
            
            var value = Int(minimum) + Int(lv*gap)
            levelList.append(value)
        }
        levelList.append(Int(maximum))
       return levelList
    }
    
    func genLevelHVs(totalLevel: Int, D: Int) -> [Int: [Int]]{
        
        print("generating level HVs")
        var levelHVs: [Int: [Int]] = [:]
        var indexVector = 1...D
        var nextLevel = Int(D/2/totalLevel)
        var change = Int(D / 2)
        var toOne: [Int] = []
        for level in stride(from:0, to: totalLevel, by: 1){
            var name = level
            
            var base = Array(repeating: -1, count: D)

            if level == 0{
                
                for _ in 1..<D {
                    let randomNumber = Int.random(in: 1...D)
                    toOne.append(randomNumber)
                }
                
                toOne.shuffle()
                toOne = Array(toOne[..<change])
            }
            else{
                for _ in 1..<D {
                    let randomNumber = Int.random(in: 1...D)
                    toOne.append(randomNumber)
                }
                
                toOne.shuffle()
                toOne = Array(toOne[..<nextLevel])
                
                
            }
           
            for index in toOne{
                base[index] = base[index] * -1
            }
            levelHVs[name] = base
        }
        
        
        return levelHVs
    }
    
    func genIDHVs(totalPos: NSNumber, D: Int) -> [Int: [Int]]{
        print("generating ID HVs")
        var IDHVs: [Int: [Int]] = [:]
        var indexVector = D
        var change = Int(D / 2)
        var toOne: [Int] = []
        
        for level in stride(from:0, to: totalLevel, by: 1){
        var name = level
        var base = Array(repeating: -1, count: D)
        
            for _ in 1..<D {
                let randomNumber = Int.random(in: 1...D)
                toOne.append(randomNumber)
            }
            
            toOne.shuffle()
            toOne = Array(toOne[..<change])
            
            for index in toOne{
                base[index] = 1
            }
            IDHVs[name] = base
            
        }
        return IDHVs
    }
    
    }
