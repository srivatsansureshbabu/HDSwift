//
//  main.swift
//  HDSwiftProject
//
//  Created by Srivatsan Suresh Babu on 7/18/24.
//
import Foundation
import CoreML
print("hello world")

func readLastColumnAsIntegers(from csvFile: String) -> [Int] {
    // Get the file URL from the app bundle
    guard let fileURL = Bundle.main.url(forResource: csvFile, withExtension: "csv") else {
        fatalError("File not found")
    }
    
    do {
        // Read the contents of the file into a string
        let csvData = try String(contentsOf: fileURL, encoding: .utf8)
        
        // Split the CSV data into rows based on newlines
        let rows = csvData.split(separator: "\n").filter { !$0.isEmpty }
        
        // Initialize an array to store the last column values as integers
        var lastColumn = [Int]()
        
        // Iterate over each row
        for (index, row) in rows.enumerated() {
            // Split the row into columns based on commas
            let columns = row.split(separator: ",")
            
            // Check if the row has at least one column
            if let lastString = columns.last {
                let trimmedString = lastString.trimmingCharacters(in: .whitespacesAndNewlines)
                
                // Attempt to convert the last column value to an integer
                if let lastInt = Int(trimmedString) {
                    // Add the integer value to the lastColumn array
                    lastColumn.append(lastInt)
                } else {
                    // Handle the case where the conversion to integer fails
                    fatalError("Conversion to integer failed for value: '\(trimmedString)' in row \(index + 1)")
                }
            }
        }
        
        // Return the array of last column values as integers
        return lastColumn
    } catch {
        // Handle any errors that occur during file reading
        fatalError("Error reading file: \(error)")
    }
}


func readAllRowsExceptTopAsDoubles(from csvFile: String) -> [[Double]] {
    // Get the file URL from the app bundle
    guard let fileURL = Bundle.main.url(forResource: csvFile, withExtension: "csv") else {
        fatalError("File not found")
    }
    
    do {
        // Read the contents of the file into a string
        let csvData = try String(contentsOf: fileURL, encoding: .utf8)
        
        // Split the CSV data into rows based on newlines
        let rows = csvData.split(separator: "\n")
        
        // Check if there are at least two rows (one header and at least one data row)
        guard rows.count > 1 else {
            fatalError("CSV file does not contain enough rows")
        }
        
        // Initialize an array to store the processed rows as arrays of doubles
        var result = [[Double]]()
        
        // Skip the first row (header) and process the remaining rows
        for (index, row) in rows.dropFirst().enumerated() {
            // Split the row into columns based on commas and convert each column to a double
            let doubleRow = row.split(separator: ",").compactMap { column -> Double? in
                let trimmedColumn = column.trimmingCharacters(in: .whitespacesAndNewlines)
                if let value = Double(trimmedColumn) {
                    return value
                } else {
                    print("Conversion to double failed for value: \(trimmedColumn) in row \(index + 2)")
                    return nil
                }
            }
            
            // Ensure the entire row was successfully converted before adding it to the result
            if doubleRow.count == row.split(separator: ",").count {
                result.append(doubleRow)
            } else {
                fatalError("Failed to convert entire row \(index + 2) to doubles")
            }
        }
        
        // Return the array of rows as arrays of doubles
        return result
    } catch {
        // Handle any errors that occur during file reading
        fatalError("Error reading file: \(error)")
    }
}


// Usage example
var testLabels = readLastColumnAsIntegers(from: "isolet_testInt")
var trainLabels = readLastColumnAsIntegers(from: "isolet_trainInt")
var testData = readAllRowsExceptTopAsDoubles(from: "isolet_testInt")
var trainData = readAllRowsExceptTopAsDoubles(from: "isolet_trainInt")

// Example predefined train data
//let trainData: [[Double]] = [
//    [0.1, 0.2, 0.3, 0.4, 0.5],
//    [0.6, 0.7, 0.8, 0.9, 1.0],
//    [0.2, 0.3, 0.4, 0.5, 0.6],
//    [0.7, 0.8, 0.9, 1.0, 0.1],
//    [0.3, 0.4, 0.5, 0.6, 0.7],
//    [0.4, 0.5, 0.6, 0.7, 0.8],
//    [0.5, 0.6, 0.7, 0.8, 0.9],
//    [0.8, 0.9, 1.0, 0.1, 0.2],
//    [0.9, 1.0, 0.1, 0.2, 0.3],
//    [1.0, 0.1, 0.2, 0.3, 0.4],
//    [0.3, 0.4, 0.5, 0.6, 0.7],
//    [0.4, 0.5, 0.6, 0.7, 0.8],
//    [0.5, 0.6, 0.7, 0.8, 0.9],
//    [0.6, 0.7, 0.8, 0.9, 1.0],
//    [0.7, 0.8, 0.9, 1.0, 0.1],
//    [0.8, 0.9, 1.0, 0.1, 0.2],
//    [0.9, 1.0, 0.1, 0.2, 0.3],
//    [1.0, 0.1, 0.2, 0.3, 0.4],
//    [0.2, 0.3, 0.4, 0.5, 0.6],
//    [0.3, 0.4, 0.5, 0.6, 0.7],
//    [0.4, 0.5, 0.6, 0.7, 0.8],
//    [0.5, 0.6, 0.7, 0.8, 0.9],
//    [0.6, 0.7, 0.8, 0.9, 1.0],
//    [0.7, 0.8, 0.9, 1.0, 0.1],
//    [0.8, 0.9, 1.0, 0.1, 0.2],
//    [0.9, 1.0, 0.1, 0.2, 0.3]
//]
//
//// Example predefined train labels
//let trainLabels: [Int] = Array(1...26)
//
//// Example predefined test data
//let testData: [[Double]] = [
//    [0.5, 0.4, 0.3, 0.2, 0.1],
//    [1.0, 0.9, 0.8, 0.7, 0.6],
//    [0.6, 0.5, 0.4, 0.3, 0.2],
//    [0.7, 0.6, 0.5, 0.4, 0.3],
//    [0.8, 0.7, 0.6, 0.5, 0.4],
//    [0.9, 0.8, 0.7, 0.6, 0.5],
//    [1.0, 0.9, 0.8, 0.7, 0.6],
//    [0.1, 0.2, 0.3, 0.4, 0.5],
//    [0.2, 0.3, 0.4, 0.5, 0.6],
//    [0.3, 0.4, 0.5, 0.6, 0.7]
//]

// Example predefined test labels
//let testLabels: [Int] = Array(1...10)

let D = 10000
let nLevels = 100
let n = 10
var model = HDModel(trainData: trainData, trainLabels: trainLabels, testData: testData, testLabels: testLabels, D: D, totalLevel: nLevels)

let levelList = model.getLevelList( trainData: trainData, totalLevel: nLevels)

print("YOOOOOP")
print(type(of: testData))
