import Foundation

struct IsoletData: Codable {
    let trainData: TrainData
    let testData: TestData
}

struct TrainData: Codable {
    let features: [String: [Float]]
    let labels: [Int]
}

struct TestData: Codable {
    let features: [String: [Float]]
    let labels: [Int]
}
