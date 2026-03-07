import Foundation
import AVFoundation

final class AudioRecorderService: NSObject {
    private var recorder: AVAudioRecorder?
    private let meterQueue = DispatchQueue(label: "com.aether.audio-recorder.meter")
    private var meterTimer: Timer?

    var onPowerLevel: ((CGFloat) -> Void)?
    var onDuration: ((TimeInterval) -> Void)?

    static func requestMicrophonePermission() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            return await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        @unknown default:
            return false
        }
    }

    func startRecording() throws {
        stopRecording(discard: true)

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)

        let outputURL = Self.makeOutputURL()
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 22050,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.medium.rawValue,
        ]

        let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
        recorder.isMeteringEnabled = true
        recorder.prepareToRecord()
        guard recorder.record() else {
            throw NSError(domain: "AudioRecorderService", code: 1, userInfo: [NSLocalizedDescriptionKey: "failed to start recording"])
        }

        self.recorder = recorder
        startMetering()
    }

    func stopRecording(discard: Bool = false) {
        meterTimer?.invalidate()
        meterTimer = nil
        recorder?.stop()
        if discard, let url = recorder?.url {
            try? FileManager.default.removeItem(at: url)
        }
        recorder = nil
        onPowerLevel?(0)
        onDuration?(0)
    }

    func finishRecording() -> URL? {
        let output = recorder?.url
        stopRecording(discard: false)
        return output
    }

    var isRecording: Bool {
        recorder?.isRecording ?? false
    }

    private func startMetering() {
        meterTimer?.invalidate()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
            guard let self, let recorder = self.recorder, recorder.isRecording else { return }
            self.meterQueue.async {
                recorder.updateMeters()
                let avgPower = recorder.averagePower(forChannel: 0)
                let normalized = Self.normalizePower(avgPower)
                let duration = recorder.currentTime
                DispatchQueue.main.async {
                    self.onPowerLevel?(normalized)
                    self.onDuration?(duration)
                }
            }
        }
    }

    private static func normalizePower(_ power: Float) -> CGFloat {
        if power <= -80 { return 0.0 }
        if power >= 0 { return 1.0 }
        return CGFloat((power + 80) / 80)
    }

    private static func makeOutputURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("voice-\(UUID().uuidString)")
            .appendingPathExtension("m4a")
    }
}
