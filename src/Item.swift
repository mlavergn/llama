//
//  Item.swift
//  llama3
//
//  Created by Marc Lavergne on 6/13/24.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
