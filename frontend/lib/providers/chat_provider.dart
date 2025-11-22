import 'package:flutter/foundation.dart';

import '../models/chat_message.dart';
import '../services/chat_agent_service.dart';
import '../services/api_client.dart';

class ChatProvider extends ChangeNotifier {
  ChatProvider();

  final List<ChatMessage> _messages = [];
  bool _isTyping = false;
  String? _caseId;
  List<String> _followups = [];
  ChatAgentService _service = ChatAgentService();
  final ApiClient _apiClient = ApiClient();
  String? _pendingUploadType; // 'lab_pdf' | 'image' | 'vitals'

  List<ChatMessage> get messages => List.unmodifiable(_messages);
  bool get isTyping => _isTyping;
  String? get caseId => _caseId;
  List<String> get followups => List.unmodifiable(_followups);
  String? get pendingUploadType => _pendingUploadType;

  Future<void> sendMessage(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return;

    _messages.add(
      ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        sender: ChatSender.user,
        text: trimmed,
        timestamp: DateTime.now(),
      ),
    );
    notifyListeners();

    _isTyping = true;
    notifyListeners();

    try {
      final result = await _service.sendMessage(caseId: _caseId, message: trimmed);

      _caseId = result.caseId.isEmpty ? _caseId : result.caseId;
      _followups = result.action.followups;

      // If agent requests an upload, set pendingUploadType so UI can trigger picker.
      if (result.action.action == 'request_upload') {
        _pendingUploadType = result.action.uploadType;
      }

      if (result.reply.isNotEmpty) {
        _messages.add(
          ChatMessage(
            id: 'doctor-${DateTime.now().millisecondsSinceEpoch}',
            sender: ChatSender.doctor,
            text: result.reply,
            timestamp: DateTime.now(),
            metadata: result.action.metadata,
          ),
        );
      }
    } catch (e) {
      _messages.add(
        ChatMessage(
          id: 'error-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Sorry, I could not contact the medical agent: $e',
          timestamp: DateTime.now(),
        ),
      );
    } finally {
      _isTyping = false;
      notifyListeners();
    }
  }

  void consumePendingUploadType() {
    _pendingUploadType = null;
    notifyListeners();
  }

  Future<void> uploadLabReport(String filePath) async {
    if (_caseId == null) {
      _messages.add(
        ChatMessage(
          id: 'system-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Please describe your symptoms first so I can create a case before attaching lab reports.',
          timestamp: DateTime.now(),
        ),
      );
      notifyListeners();
      return;
    }

    _isTyping = true;
    notifyListeners();

    try {
      await _apiClient.uploadLabReport(caseId: _caseId!, filePath: filePath);
      _messages.add(
        ChatMessage(
          id: 'system-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Your lab report has been uploaded for this case.',
          timestamp: DateTime.now(),
        ),
      );
    } catch (e) {
      _messages.add(
        ChatMessage(
          id: 'error-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Failed to upload lab report: $e',
          timestamp: DateTime.now(),
        ),
      );
    } finally {
      _isTyping = false;
      notifyListeners();
    }
  }

  Future<void> uploadImaging(String filePath) async {
    if (_caseId == null) {
      _messages.add(
        ChatMessage(
          id: 'system-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Please describe your symptoms first so I can create a case before attaching imaging.',
          timestamp: DateTime.now(),
        ),
      );
      notifyListeners();
      return;
    }

    _isTyping = true;
    notifyListeners();

    try {
      await _apiClient.uploadImage(caseId: _caseId!, filePath: filePath);
      _messages.add(
        ChatMessage(
          id: 'system-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Your imaging file has been uploaded for this case.',
          timestamp: DateTime.now(),
        ),
      );
    } catch (e) {
      _messages.add(
        ChatMessage(
          id: 'error-${DateTime.now().millisecondsSinceEpoch}',
          sender: ChatSender.system,
          text: 'Failed to upload imaging: $e',
          timestamp: DateTime.now(),
        ),
      );
    } finally {
      _isTyping = false;
      notifyListeners();
    }
  }
}

