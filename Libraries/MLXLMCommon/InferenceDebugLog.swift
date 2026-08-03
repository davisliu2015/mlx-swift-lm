import Foundation

/// 推理调试日志钩子：由宿主 App（如 smlx）在启动时注入 handler，
/// fork 内部关键事件（停止原因、MTP 迭代器异常退出等）经此上报。
/// 不注入则为空操作，fork 自身不做任何 IO。
public enum InferenceDebugLog {
    nonisolated(unsafe) public static var handler: (@Sendable (String) -> Void)?

    @inline(__always)
    static func log(_ message: @autoclosure () -> String) {
        guard let handler else { return }
        handler(message())
    }
}
