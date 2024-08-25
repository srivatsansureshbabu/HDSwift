import UIKit
import CoreML
import CreateMLComponents
import Foundation
import Accelerate
import Dispatch

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
    
    func buildBufferHVs(modelMultiply: ElementWiseMultiplication617x10, modelAdd: ElementWiseAddition_2D_to_1D_617x10 , mode: String, D: Int, dataset: String){
        
        if mode == "train"{
            
            print("Encoding Training Data")
//            for i in 0..<self.trainData.count{
//                self.trainHVs.append(IDMultHV(modelMultiply: modelMultiply, modelAdd: modelAdd, inputBuffer: self.trainData[i], D: D, levelHVs: self.levelHVs, levelList: self.levelList, IDHVs: self.IDHVs)!)
//                print( Float(i)/Float(self.trainData.count) )
//            }
            
            let semaphore = DispatchSemaphore(value: 4)
            DispatchQueue.concurrentPerform(iterations: self.trainData.count) { i in
                semaphore.wait()
                
                self.trainHVs.append(IDMultHV(modelMultiply: modelMultiply, modelAdd: modelAdd, inputBuffer: self.trainData[i], D: D, levelHVs: self.levelHVs, levelList: self.levelList, IDHVs: self.IDHVs)!)
                print( Float(i)/Float(self.trainData.count) )
                
                semaphore.signal()
            }
            
            self.trainHVs = binarize(array: self.trainHVs)
            
            self.classHVs = oneHVPerClass(inputLabels: self.trainLabels, inputHVs: self.trainHVs, D: self.D)!
            //            self.classHVs = convertMLMultiArrayTo2DArray(val!)!
            
            
        }
        else{
            print("Encoding Testing Data")
            
//            for index in 0..<testData.count{
//                self.testHVs.append(IDMultHV(modelMultiply: modelMultiply, modelAdd: modelAdd, inputBuffer: self.testData[index], D: self.D, levelHVs: self.levelHVs, levelList: self.levelList, IDHVs: self.IDHVs)!)
//                print( Float(index)/Float(self.testData.count) )
//                
//            }
            
            let semaphore = DispatchSemaphore(value: 4)
            
            DispatchQueue.concurrentPerform(iterations: testData.count) { index in
                semaphore.wait()
                
                self.testHVs.append(IDMultHV(modelMultiply: modelMultiply, modelAdd: modelAdd, inputBuffer: self.testData[index], D: self.D, levelHVs: self.levelHVs, levelList: self.levelList, IDHVs: self.IDHVs)!)
                print( Float(index)/Float(self.testData.count) )
                
                semaphore.signal()
            }
            
            
            self.testHVs = binarize(array: self.testHVs)
        }
    }
    
//    func IDMultHV(modelMultiply: ElementWiseMultiplication617x10, modelAdd: ElementWiseAddition_2D_to_1D_617x10, inputBuffer: [Float], D: Int, levelHVs: [Int: [Float]], levelList: [Float], IDHVs: [Int:[Float]]) -> [Float]?{
//
//        var sumHV = zeros(size: D)!
//        
//        for keyVal in 0..<inputBuffer.count{
//            let IDHV = IDHVs[keyVal]
//            let key = numToKey(value: inputBuffer[keyVal], levelList: levelList)
//            let levelHV = levelHVs[key]
//            
////            if keyVal < 5 {
////                print("IDHVS: \n\n")
////                print(IDHV!) // D dimensionality
////                print("levelHVs: \n\n")
////                print(levelHV!) // D dimensionality
////            }
//            
//            // element wise multiply them
//            for i in 0..<IDHV!.count {
//                sumHV[i] += IDHV![i] * levelHV![i]
//            }
//            
//            
//            // is there a way we could make IDHV and levelHV into a 2D array and then do operations based on that, and put them into a 1D array?
//        }
//        
//        return sumHV
//    }
    
    
    
    func IDMultHV(modelMultiply: ElementWiseMultiplication617x10, modelAdd: ElementWiseAddition_2D_to_1D_617x10 , inputBuffer: [Float], D: Int, levelHVs: [Int: [Float]], levelList: [Float], IDHVs: [Int:[Float]]) -> [Float]?{
    
            var sumHV = zeros(size: D)!
    
            let IDHVs2D: [[Float]] = Array(IDHVs.values) // converts IDHVs into a 2D array
            var levelHVs2D: [[Float]] = Array(IDHVs.values) // converts levelHVs into a 2D array
    
            // well maybe we could do element-wise multiplication between a 617x10, then it will be a element wise addition of that 617x10 to a 1x10 array...
        
            for keyVal in 0..<inputBuffer.count{
                let key = numToKey(value: inputBuffer[keyVal], levelList: levelList)
                levelHVs2D[keyVal] = levelHVs[key]!
            }
            
            // element-wise multiplies and adds it into sumHV
            
             var elementWiseMultiplied = performElementWiseMultiplication(model: modelMultiply, inputArray1: IDHVs2D, inputArray2: levelHVs2D)

        for i in 0..<D{
            for j in 0..<elementWiseMultiplied!.count{
                sumHV[i] += elementWiseMultiplied![j][i]
            }
        }
             
//                for j in 0..<IDHVs2D.count{
//    
//                    for k in 0..<IDHVs2D[j].count{
//                        sumHV[k] += IDHVs2D[j][k] * levelHVs2D[j][k]
//                    }
//                    print(sumHV)
//                }
    
                // is there a way we could make IDHV and levelHV into a 2D array and then do operations based on that, and put them into a 1D array?
            return sumHV
        }
    
    
    
//        func IDMultHV(modelMultiply: ElementWiseMultiplication617x10, modelAdd: ElementWiseAddition_2D_to_1D_617x10 , inputBuffer: [Float], D: Int, levelHVs: [Int: [Float]], levelList: [Float], IDHVs: [Int:[Float]]) -> [Float]?{
//    
////                var sumHV = zeros(size: D)!
//    
//                let IDHVs2D: [[Float]] = Array(IDHVs.values) // converts IDHVs into a 2D array
//                var levelHVs2D: [[Float]] = Array(IDHVs.values) // converts levelHVs into a 2D array
//    
//                // well maybe we could do element-wise multiplication between a 617x10, then it will be a element wise addition of that 617x10 to a 1x10 array...
//    
//                for keyVal in 0..<inputBuffer.count{
//                    let key = numToKey(value: inputBuffer[keyVal], levelList: levelList)
//                    levelHVs2D[keyVal] = levelHVs[key]!
//                }
//    
//                // element-wise multiplies and adds it into sumHV
//            
//            let sumHV = performElementWiseMultiplication(model: modelMultiply, inputArray1: IDHVs2D, inputArray2: levelHVs2D)
////            for i in 0..<D{
////                for j in 0..<elementWiseMultiplied!.count{
////                    sumHV[i] += elementWiseMultiplied![j][i]
////                }
////            }
//    
//    //                for j in 0..<IDHVs2D.count{
//    //
//    //                    for k in 0..<IDHVs2D[j].count{
//    //                        sumHV[k] += IDHVs2D[j][k] * levelHVs2D[j][k]
//    //                    }
//    //                    print(sumHV)
//    //                }
//    
//                    // is there a way we could make IDHV and levelHV into a 2D array and then do operations based on that, and put them into a 1D array?
//                return sumHV
//            }
    
    
    
    
    
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
    
    
    func oneHVPerClass(inputLabels: [Int], inputHVs: [[Float]], D: Int) -> [[Float]]?{
        let numClasses = (inputLabels.max() ?? 0) + 1
        
        //         creates 2D multiarray [numClasses, D]
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
        
        
        return convertMLMultiArrayTo2DArray(classHVs)
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
                
                // element wise addition
                //                retClassHVs[trainLabels[index]] = performElementWiseAddition(model: model, inputArray1: retClassHVs[trainLabels[index]], inputArray2: trainHVs[index])!
                
                //                classHVs_binary[guess] = binarize(array: retClassHVs[guess])
                //                classHVs_binary[trainLabels[index]] = binarize(array: retClassHVs[trainLabels[index]])
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
    
    static func buildHDModel(modelMultiply: ElementWiseMultiplication617x10, modelAdd: ElementWiseAddition_2D_to_1D_617x10, trainData: [[Float]], trainLabels: [Int], testData: [[Float]], testLabels: [Int], D: Int, nLevels: Int, datasetName: String) -> HDModel{
        let model = HDModel(trainData: trainData, trainLabels: trainLabels, testData: testData, testLabels: testLabels, D: D, totalLevel: nLevels)
        model.buildBufferHVs(modelMultiply: modelMultiply, modelAdd: modelAdd, mode: "train", D: D, dataset: datasetName)
        model.buildBufferHVs(modelMultiply: modelMultiply, modelAdd: modelAdd, mode: "test", D: D, dataset: datasetName)
        
        return model
    }
    
    
    
    
    
    
    // Define a function to perform element-wise addition
//    func performElementWiseAddition(model: ElementwiseAdd10000, inputArray1: [Float], inputArray2: [Float]) -> [Float]? {
//        //         Load the model
//        
//        guard let input1 = createMLMultiArray(from: inputArray1, shape: [1, inputArray1.count]),
//              let input2 = createMLMultiArray(from: inputArray2, shape: [1, inputArray2.count]) else {
//            fatalError("Failed to create input arrays")
//        }
//        
//        // Perform prediction
//        do {
//            let prediction = try model.prediction(input1: input1, input2: input2)
//            
//            // Access and print the result using the output name 'Identity'
//            if let result = prediction.Identity as? MLMultiArray { // 'Identity' is the output name
//                if let resultArray = multiArrayToFloatArray(result) {
//                    //                    print("Result Array: \(resultArray)")
//                    return resultArray
//                } else {
//                    print("Failed to convert result to array")
//                    return nil
//                    
//                }
//            }
//        } catch {
//            print("Error making prediction: \(error)")
//            return nil
//        }
//        
//        
//    }
    func performElementWiseAddition(model: ElementWiseAddition_2D_to_1D_617x10, inputArray1: [[Float]]) -> [Float]? {
        //         Load the model
        
        guard let input = createMLMultiArray2D(from: inputArray1) else {
            fatalError("Failed to create input arrays")
        }
        
        guard let input1 = convert2DArrayTo3DMLMultiArray(array: inputArray1) else{
            fatalError("Not working")
        }
        // Perform prediction
        do {
            let prediction = try model.prediction(input_array: input1)
            
            // Access and print the result using the output name 'Identity'
            if let result = prediction.Identity as? MLMultiArray { // 'Identity' is the output name
                if let resultArray = multiArrayToFloatArray(result) {
                    //                    print("Result Array: \(resultArray)")
                    return resultArray
                } else {
                    print("Failed to convert result to array")
                    return nil
                    
                }
            }
        } catch {
            print("Error making prediction: \(error)")
            return nil
        }
        
        
    }
    
    

    
    func performElementWiseMultiplication(model: ElementWiseMultiplication617x10, inputArray1: [[Float]], inputArray2: [[Float]]) -> [[Float]]? { // [[Float]] ?
        //         Load the model
        
        guard let input1 = convert2DArrayTo3DMLMultiArray(array: inputArray1),
              let input2 = convert2DArrayTo3DMLMultiArray(array: inputArray2) else {
            fatalError("Failed to create input arrays")
        }
        
        // Perform prediction
        do {
            let prediction = try model.prediction(input1: input1, input2: input2)
            
            // Access and print the result using the output name 'Identity'
            if let result = prediction.Identity as? MLMultiArray { // 'Identity' is the output name
                if let resultArray = convert3DMLMultiArrayTo2DArray(mlMultiArray: result) {
                    //                    print("Result Array: \(resultArray)")
                    return resultArray
                } else {
                    print("Failed to convert result to array")
                    return nil
                    
                }
            }
        } catch {
            print("Error making prediction: \(error)")
            return nil
        }
        
        
    }
    
    func convert3DMLMultiArrayTo2DArray(mlMultiArray: MLMultiArray) -> [[Float]]? {
        // Ensure the MLMultiArray has the expected shape
        guard mlMultiArray.shape.count == 3 else {
            print("Expected 3D MLMultiArray.")
            return nil
        }
        
        // Extract the dimensions
        let batchSize = mlMultiArray.shape[0].intValue
        let height = mlMultiArray.shape[1].intValue
        let width = mlMultiArray.shape[2].intValue
        
        guard batchSize == 1 else {
            print("Expected batch size of 1.")
            return nil
        }
        
        // Create a 2D array to hold the result
        var array: [[Float]] = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        
        // Fill the 2D array with values from the 3D MLMultiArray
        for i in 0..<height {
            for j in 0..<width {
                let value = mlMultiArray[[0, NSNumber(value: i), NSNumber(value: j)]].floatValue ?? 0.0
                array[i][j] = value
            }
        }
        
        return array
    }
    
    
    func convert2DArrayTo3DMLMultiArray(array: [[Float]]) -> MLMultiArray? {
        let height = array.count // Number of rows
        guard height > 0 else { return nil } // Ensure the array is not empty
        let width = array[0].count // Number of columns
        
        // Create a 3D MLMultiArray with shape [1, height, width]
        let shape: [NSNumber] = [1, NSNumber(value: height), NSNumber(value: width)]
        
        do {
            let mlMultiArray = try MLMultiArray(shape: shape, dataType: .float32)
            
            // Fill the 3D MLMultiArray with the 2D array data
            for i in 0..<height {
                for j in 0..<width {
                    mlMultiArray[[0, NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: array[i][j])
                }
            }
            
            return mlMultiArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }
    
    
    
    
    
    
    
    // Function to create MLMultiArray from [[Float]] 2D array
    func createMLMultiArray2D(from array: [[Float]]) -> MLMultiArray? {
        // Get the shape dimensions (should be 10x10)
        let rows = array.count
        let columns = array.first?.count ?? 0
        
        // Ensure the input array has a consistent shape
        guard rows == 617, columns == 10 else {
            print("Input array must have a 617x10 shape.")
            return nil
        }
        
        // Create an MLMultiArray with the given shape
        let shape: [NSNumber] = [NSNumber(value: rows), NSNumber(value: columns)]
        guard let mlArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            print("Failed to create MLMultiArray.")
            return nil
        }
        
        // Fill the MLMultiArray with values from the [[Float]] array
        for i in 0..<rows {
            for j in 0..<columns {
                mlArray[[i as NSNumber, j as NSNumber]] = NSNumber(value: array[i][j])
            }
        }
        
        return mlArray
    }
    
    
    
    // Convert MLMultiArray to [Float]
    func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float]? {
        // Check if the MLMultiArray has the right data type
        guard multiArray.dataType == .float32 else {
            print("Unsupported data type")
            return nil
        }
        
        // Convert MLMultiArray to [Float] using a more direct approach
        let count = multiArray.count
        var array = [Float](repeating: 0, count: count)
        
        for i in 0..<count {
            // Use `floatValue` to access the values
            array[i] = multiArray[i].floatValue
        }
        
        return array
    }
    
    // Helper function to create MLMultiArray from array with specified shape
    func createMLMultiArray(from array: [Float], shape: [Int]) -> MLMultiArray? {
        do {
            let multiArray = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            for (index, value) in array.enumerated() {
                multiArray[index] = NSNumber(value: value)
            }
            return multiArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
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
            //            print(value)
            return nil
        }
    }
    
    func convertMLMultiArrayTo2DArray(_ multiArray: MLMultiArray) -> [[Float]]? {
        // Check if the MLMultiArray is 2D
        guard multiArray.shape.count == 2 else {
            print("The MLMultiArray is not 2D.")
            return nil
        }
        
        let rows = multiArray.shape[0].intValue
        let columns = multiArray.shape[1].intValue
        
        // Initialize a 2D array with the appropriate dimensions
        var result = [[Float]](repeating: [Float](repeating: 0.0, count: columns), count: rows)
        
        // Extract data from MLMultiArray
        let dataPointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
        
        // Populate the 2D array
        for row in 0..<rows {
            for column in 0..<columns {
                let index = row * columns + column
                result[row][column] = dataPointer[index]
            }
        }
        
        return result
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
