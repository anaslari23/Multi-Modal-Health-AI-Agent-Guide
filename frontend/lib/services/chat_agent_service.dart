import 'dart:convert';

import 'package:http/http.dart' as http;

import 'api_client.dart';

class AgentActionDto {
  AgentActionDto({
    required this.action,
    required this.text,
    required this.followups,
    required this.uploadType,
    required this.metadata,
  });

  final String action;
  final String? text;
  final List<String> followups;
  final String? uploadType;
  final Map<String, dynamic> metadata;

  factory AgentActionDto.fromJson(Map<String, dynamic> json) {
    return AgentActionDto(
      action: json['action'] as String? ?? 'info',
      text: json['text'] as String?,
      followups: (json['followups'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          const [],
      uploadType: json['upload_type'] as String?,
      metadata: (json['metadata'] as Map<String, dynamic>?) ?? <String, dynamic>{},
    );
  }
}

class ChatAgentResult {
  ChatAgentResult({
    required this.reply,
    required this.caseId,
    required this.action,
    this.diagnosis,
  });

  final String reply;
  final String caseId;
  final AgentActionDto action;
  final Map<String, dynamic>? diagnosis;
}

class ChatAgentService {
  ChatAgentService({this.baseUrl = kApiBaseUrl});

  final String baseUrl;

  Future<ChatAgentResult> sendMessage({
    String? caseId,
    required String message,
  }) async {
    final uri = Uri.parse('$baseUrl/agent/chat');
    final res = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'case_id': caseId,
        'message': message,
      }),
    );

    if (res.statusCode != 200) {
      throw Exception('Agent chat failed: ${res.statusCode} ${res.body}');
    }

    final decoded = jsonDecode(res.body) as Map<String, dynamic>;
    final action = AgentActionDto.fromJson(decoded['action'] as Map<String, dynamic>);
    final diagnosis = decoded['diagnosis'] as Map<String, dynamic>?;

    return ChatAgentResult(
      reply: decoded['reply']?.toString() ?? '',
      caseId: decoded['case_id']?.toString() ?? '',
      action: action,
      diagnosis: diagnosis,
    );
  }
}

