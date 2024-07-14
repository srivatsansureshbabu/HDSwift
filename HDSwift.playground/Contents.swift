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
    
    func genLevelHVs(totalLevel: Int, D: Int) -> [String: Int]{
        
        print("generating level HVs")
        var levelHVs: [String: Int] = [:]
        var indexVector = D
        var nextLevel = Int(D/2/totalLevel)
        var change = Int(D / 2)
        var toOne: [Int] = []
        for level in stride(from:0, to: totalLevel, by: 1){
            var name = level
            
            if level == 0{
                let base = createMLMultiArray(dimension: D, baseVal: -1)
                let toOne = MLShuffleArray(base!);
//                 toOne = toOne(indexVector)[:change]
            }
            else{
                let toOne = createMLMultiArray(dimension: D, baseVal: -1)
//                toOne = toOne(indexVector)[:nextLevel]
            }
            for index in toOne{
                // base[index] = base[index] * -1
                // levelHVs[name] = copy.deepcopy(base)
            }
        }
        
        return levelHVs
    }
    
//    func genIDHVs(totalPos: NSNumber, D: Int) -> [String: Int]{
//        print("generating ID HVs")
//        let IDHVs: [String: Int] = [:]
//        let indexVector = D
//        let change = Int(D / 2)
//        let toOne: [Int] = []
//        
//        return ["Hello": 10]
//    }
    
    }
