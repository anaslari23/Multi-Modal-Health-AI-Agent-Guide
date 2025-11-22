import 'package:flutter/material.dart';

import '../../models/chat_message.dart';

class ChatBubble extends StatelessWidget {
  const ChatBubble({
    super.key,
    required this.message,
  });

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final isUser = message.sender == ChatSender.user;
    final colorScheme = Theme.of(context).colorScheme;
    final metadata = message.metadata;
    
    // Extract drug suggestions if available
    List<dynamic>? drugSuggestions;
    if (metadata != null && metadata.containsKey('diagnosis')) {
      final diag = metadata['diagnosis'];
      if (diag is Map && diag.containsKey('drug_suggestions')) {
        drugSuggestions = diag['drug_suggestions'];
      }
    }

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
        decoration: BoxDecoration(
          color: isUser
              ? colorScheme.primary
              : colorScheme.surfaceVariant.withOpacity(0.8),
          borderRadius: BorderRadius.circular(18).copyWith(
            bottomLeft: Radius.circular(isUser ? 18 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 18),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              message.text,
              style: TextStyle(
                color: isUser ? colorScheme.onPrimary : colorScheme.onSurface,
              ),
            ),
            if (drugSuggestions != null && drugSuggestions.isNotEmpty) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.medical_services, size: 16, color: colorScheme.onErrorContainer),
                        const SizedBox(width: 4),
                        Text(
                          'Medication Safety Check',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                            color: colorScheme.onErrorContainer,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    ...drugSuggestions.map((d) {
                      final safe = d['safe'] == true;
                      final name = d['name'] ?? 'Unknown';
                      final rationale = d['rationale'] ?? '';
                      return Padding(
                        padding: const EdgeInsets.only(top: 4.0),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(
                              safe ? Icons.check_circle : Icons.warning,
                              size: 14,
                              color: safe ? Colors.green : colorScheme.error,
                            ),
                            const SizedBox(width: 4),
                            Expanded(
                              child: Text(
                                '$name: $rationale',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: colorScheme.onErrorContainer,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

