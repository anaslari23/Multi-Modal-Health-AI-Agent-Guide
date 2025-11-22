enum ChatSender {
  user,
  doctor,
  system,
}

class ChatAttachment {
  ChatAttachment({
    required this.type,
    required this.label,
    this.localPath,
    this.remoteUrl,
  });

  final String type; // e.g. 'lab_pdf', 'imaging', 'vitals'
  final String label;
  final String? localPath;
  final String? remoteUrl;
}

class ChatMessage {
  ChatMessage({
    required this.id,
    required this.sender,
    required this.text,
    required this.timestamp,
    this.attachments = const [],
    this.isStreaming = false,
    this.metadata,
  });

  final String id;
  final ChatSender sender;
  final String text;
  final DateTime timestamp;
  final List<ChatAttachment> attachments;
  final bool isStreaming;
  final Map<String, dynamic>? metadata;
}

