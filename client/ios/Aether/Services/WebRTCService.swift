import Foundation
import AVFoundation
import WebRTC

/// WebRTC service for voice-to-voice communication with the Aether agent.
///
/// Replaces the old WebSocket + base64 audio approach with native WebRTC:
/// - Audio flows via WebRTC audio tracks (bidirectional, raw PCM)
/// - Text/events flow via a data channel (same JSON protocol as before)
/// - Signaling via HTTP: POST /api/webrtc/offer, PATCH /api/webrtc/ice
///
/// The server's SmallWebRTCTransport (aiortc) handles the other end.
class WebRTCService: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var connectionState: RTCIceConnectionState = .new
    @Published var isConnected = false

    // MARK: - Callbacks

    /// Called when a JSON message arrives on the data channel
    var onMessage: (([String: Any]) -> Void)?
    /// Called when connection state changes
    var onConnectionStateChange: ((RTCIceConnectionState) -> Void)?
    /// Called when data channel readiness changes
    var onDataChannelReady: ((Bool) -> Void)?

    // MARK: - Private

    private var peerConnection: RTCPeerConnection?
    private var dataChannel: RTCDataChannel?
    private var localAudioTrack: RTCAudioTrack?
    private var pcId: String?

    private let factory: RTCPeerConnectionFactory
    private let audioQueue = DispatchQueue(label: "com.aether.webrtc.audio")

    private var baseURL = "http://localhost:3080"
    private var token = ""

    /// Keep-alive ping timer
    private var pingTimer: Timer?

    var isDataChannelOpen: Bool {
        dataChannel?.readyState == .open
    }

    // MARK: - Init

    override init() {
        RTCInitializeSSL()

        // The LiveKit WebRTC-SDK always routes audio through RTCAudioSession
        // internally. We must configure it (not just AVAudioSession) and use
        // useManualAudio=true so we control exactly when the audio unit starts —
        // this prevents the race condition where the audio unit tries to start
        // before the peer connection is ready.
        let rtcAudio = RTCAudioSession.sharedInstance()
        rtcAudio.lockForConfiguration()
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playAndRecord,
                mode: .voiceChat,
                options: [.defaultToSpeaker, .allowBluetoothHFP]
            )
            try AVAudioSession.sharedInstance().setActive(true)
            try rtcAudio.setActive(true)
        } catch {
            print("[WebRTC] RTCAudioSession config error: \(error)")
        }
        rtcAudio.unlockForConfiguration()

        // useManualAudio=true: we call isAudioEnabled=true ourselves once
        // the peer connection is established and we're ready to capture.
        rtcAudio.useManualAudio = true
        rtcAudio.isAudioEnabled = false  // will be enabled in setMicEnabled(true)

        let encoderFactory = RTCDefaultVideoEncoderFactory()
        let decoderFactory = RTCDefaultVideoDecoderFactory()
        factory = RTCPeerConnectionFactory(
            encoderFactory: encoderFactory,
            decoderFactory: decoderFactory
        )
        super.init()
    }

    deinit {
        disconnect()
        RTCCleanupSSL()
    }

    // MARK: - Configure

    func configure(token: String, orchestratorURL: String) {
        self.token = token
        self.baseURL = orchestratorURL
    }

    // MARK: - Microphone Permission

    /// Request microphone access. Calls completion on the main thread.
    static func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            completion(true)
        case .denied:
            completion(false)
        case .undetermined:
            AVAudioApplication.requestRecordPermission { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        @unknown default:
            completion(false)
        }
    }

    // MARK: - Connect

    func connect() async throws {
        let iceServers = [
            RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])
        ]

        let config = RTCConfiguration()
        config.iceServers = iceServers
        config.sdpSemantics = .unifiedPlan
        config.continualGatheringPolicy = .gatherContinually

        let constraints = RTCMediaConstraints(
            mandatoryConstraints: nil,
            optionalConstraints: ["DtlsSrtpKeyAgreement": "true"]
        )

        guard let pc = factory.peerConnection(
            with: config,
            constraints: constraints,
            delegate: self
        ) else {
            throw WebRTCError.failedToCreatePeerConnection
        }
        peerConnection = pc

        // Add local audio track (mic → server). Starts disabled; enabled when streaming begins.
        addLocalAudioTrack(to: pc)

        // Create data channel (text/events — same JSON protocol as WebSocket)
        let dcConfig = RTCDataChannelConfiguration()
        dcConfig.isOrdered = true
        guard let dc = pc.dataChannel(forLabel: "aether", configuration: dcConfig) else {
            throw WebRTCError.failedToCreateDataChannel
        }
        dc.delegate = self
        dataChannel = dc

        // Create SDP offer
        let offerConstraints = RTCMediaConstraints(
            mandatoryConstraints: [
                "OfferToReceiveAudio": "true",
                "OfferToReceiveVideo": "false"
            ],
            optionalConstraints: nil
        )

        let offer = try await pc.offer(for: offerConstraints)
        try await pc.setLocalDescription(offer)

        // Send offer to server via HTTP signaling
        let answer = try await sendOffer(sdp: offer.sdp, type: "offer")
        pcId = answer.pcId

        // Set remote description (server's answer)
        let remoteDesc = RTCSessionDescription(type: .answer, sdp: answer.sdp)
        try await pc.setRemoteDescription(remoteDesc)

        print("[WebRTC] Offer/answer exchange complete, pc_id=\(answer.pcId)")
    }

    // MARK: - Disconnect

    func disconnect() {
        pingTimer?.invalidate()
        pingTimer = nil

        dataChannel?.close()
        dataChannel = nil

        localAudioTrack = nil

        peerConnection?.close()
        peerConnection = nil

        pcId = nil

        DispatchQueue.main.async {
            self.isConnected = false
            self.connectionState = .closed
        }
    }

    // MARK: - Send via Data Channel

    func send(_ dict: [String: Any]) {
        guard let dc = dataChannel, dc.readyState == .open else { return }
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        let buffer = RTCDataBuffer(data: Data(text.utf8), isBinary: false)
        dc.sendData(buffer)
    }

    // MARK: - Audio Track Setup

    private func addLocalAudioTrack(to pc: RTCPeerConnection) {
        let audioConstraints = RTCMediaConstraints(
            mandatoryConstraints: [
                // Enable echo cancellation and noise suppression
                "googEchoCancellation": "true",
                "googNoiseSuppression": "true",
                "googAutoGainControl": "true",
                "googHighpassFilter": "true"
            ],
            optionalConstraints: nil
        )
        let audioSource = factory.audioSource(with: audioConstraints)
        let audioTrack = factory.audioTrack(with: audioSource, trackId: "audio0")
        // Start disabled — enabled explicitly when user starts streaming
        audioTrack.isEnabled = false
        localAudioTrack = audioTrack

        pc.add(audioTrack, streamIds: ["stream0"])
    }

    /// Enable/disable the local mic track (mute/unmute).
    /// Must also toggle RTCAudioSession.isAudioEnabled — this is what actually
    /// starts/stops the WebRTC internal audio unit (Voice Processing IO).
    func setMicEnabled(_ enabled: Bool) {
        localAudioTrack?.isEnabled = enabled
        RTCAudioSession.sharedInstance().isAudioEnabled = enabled
        print("[WebRTC] setMicEnabled(\(enabled)) — track=\(enabled), audioUnit=\(enabled)")
    }

    // MARK: - HTTP Signaling

    private struct SignalingAnswer {
        let sdp: String
        let type: String
        let pcId: String
    }

    private func sendOffer(sdp: String, type: String) async throws -> SignalingAnswer {
        let url = URL(string: "\(baseURL)/api/webrtc/offer")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let body: [String: Any] = [
            "sdp": sdp,
            "type": type,
            "user_id": ""  // Orchestrator injects user_id from session token
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw WebRTCError.signalingFailed("Offer failed with status \(statusCode)")
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let answerSdp = json["sdp"] as? String,
              let answerType = json["type"] as? String,
              let pcId = json["pc_id"] as? String else {
            throw WebRTCError.signalingFailed("Invalid answer format")
        }

        return SignalingAnswer(sdp: answerSdp, type: answerType, pcId: pcId)
    }

    private func sendIceCandidates(_ candidates: [RTCIceCandidate]) async {
        guard let pcId = pcId else { return }

        let url = URL(string: "\(baseURL)/api/webrtc/ice")!
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let candidateList = candidates.map { c -> [String: Any] in
            var dict: [String: Any] = ["candidate": c.sdp]
            if let mid = c.sdpMid {
                dict["sdpMid"] = mid
            }
            dict["sdpMLineIndex"] = c.sdpMLineIndex
            return dict
        }

        let body: [String: Any] = [
            "pc_id": pcId,
            "candidates": candidateList
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status != 200 {
                print("[WebRTC] ICE candidate send failed: status \(status)")
            }
        } catch {
            print("[WebRTC] ICE candidate send error: \(error)")
        }
    }

    // MARK: - Keep-alive

    private func startPingTimer() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let dc = self?.dataChannel, dc.readyState == .open else { return }
            let buffer = RTCDataBuffer(data: Data("ping".utf8), isBinary: false)
            dc.sendData(buffer)
        }
    }
}

// MARK: - RTCPeerConnectionDelegate

extension WebRTCService: RTCPeerConnectionDelegate {

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {
        print("[WebRTC] Signaling state: \(stateChanged.rawValue)")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {
        print("[WebRTC] Remote stream added: \(stream.streamId)")
        // Remote audio track is automatically played by WebRTC
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {
        print("[WebRTC] Remote stream removed: \(stream.streamId)")
    }

    func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {
        print("[WebRTC] Negotiation needed")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {
        print("[WebRTC] ICE connection state: \(newState.rawValue)")
        DispatchQueue.main.async {
            self.connectionState = newState
            self.isConnected = (newState == .connected || newState == .completed)
            self.onConnectionStateChange?(newState)
        }
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {
        print("[WebRTC] ICE gathering state: \(newState.rawValue)")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {
        Task {
            await sendIceCandidates([candidate])
        }
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {
        print("[WebRTC] ICE candidates removed")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {
        print("[WebRTC] Remote data channel opened: \(dataChannel.label)")
    }
}

// MARK: - RTCDataChannelDelegate

extension WebRTCService: RTCDataChannelDelegate {

    func dataChannelDidChangeState(_ dataChannel: RTCDataChannel) {
        print("[WebRTC] Data channel state: \(dataChannel.readyState.rawValue)")
        onDataChannelReady?(dataChannel.readyState == .open)
        if dataChannel.readyState == .open {
            startPingTimer()
        } else {
            pingTimer?.invalidate()
            pingTimer = nil
        }
    }

    func dataChannel(_ dataChannel: RTCDataChannel, didReceiveMessageWith buffer: RTCDataBuffer) {
        guard let text = String(data: buffer.data, encoding: .utf8) else { return }

        // Ignore pong keep-alive
        if text == "pong" { return }

        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        DispatchQueue.main.async {
            self.onMessage?(json)
        }
    }
}

// MARK: - Errors

enum WebRTCError: LocalizedError {
    case failedToCreatePeerConnection
    case failedToCreateDataChannel
    case signalingFailed(String)

    var errorDescription: String? {
        switch self {
        case .failedToCreatePeerConnection:
            return "Failed to create WebRTC peer connection"
        case .failedToCreateDataChannel:
            return "Failed to create data channel"
        case .signalingFailed(let reason):
            return "Signaling failed: \(reason)"
        }
    }
}
