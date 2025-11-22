import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import '../intake/vitals_entry_screen.dart';

import '../../core/widgets/chat_bubble.dart';
import '../../providers/chat_provider.dart';

class AiDoctorChatScreen extends StatefulWidget {
  const AiDoctorChatScreen({super.key});

  static const routeName = '/ai-doctor-chat';

  @override
  State<AiDoctorChatScreen> createState() => _AiDoctorChatScreenState();
}

class _AiDoctorChatScreenState extends State<AiDoctorChatScreen> {
  final _controller = TextEditingController();

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // If the agent requested an upload, trigger the appropriate picker automatically.
    final chat = context.read<ChatProvider>();
    final pending = chat.pendingUploadType;
    if (pending != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) async {
        await _handlePendingUploadRequest(pending);
        if (mounted) {
          context.read<ChatProvider>().consumePendingUploadType();
        }
      });
    }
  }

  Future<void> _handlePendingUploadRequest(String type) async {
    if (!mounted) return;
    switch (type) {
      case 'lab_pdf':
        final result = await FilePicker.platform.pickFiles(
          type: FileType.custom,
          allowedExtensions: ['pdf'],
        );
        final path = result?.files.single.path;
        if (path != null && mounted) {
          await context.read<ChatProvider>().uploadLabReport(path);
        }
        break;
      case 'image':
        final picker = ImagePicker();
        final picked = await picker.pickImage(source: ImageSource.gallery);
        if (picked != null && mounted) {
          await context.read<ChatProvider>().uploadImaging(picked.path);
        }
        break;
      case 'vitals':
        if (!mounted) return;
        Navigator.pushNamed(context, VitalsEntryScreen.routeName);
        break;
      default:
        // Unknown type; show chooser instead.
        await _showAttachSheet();
    }
  }

  Future<void> _showAttachSheet() async {
    if (!mounted) return;
    final parentContext = context; // stable ancestor context
    await showModalBottomSheet<void>(
      context: parentContext,
      builder: (sheetContext) {
        return SafeArea(
          child: Wrap(
            children: [
              ListTile(
                leading: const Icon(Icons.picture_as_pdf_outlined),
                title: const Text('Attach lab report (PDF)'),
                onTap: () async {
                  Navigator.pop(sheetContext);
                  final result = await FilePicker.platform.pickFiles(
                    type: FileType.custom,
                    allowedExtensions: ['pdf'],
                  );
                  final path = result?.files.single.path;
                  if (path != null && mounted) {
                    await parentContext.read<ChatProvider>().uploadLabReport(path);
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.image_outlined),
                title: const Text('Attach imaging (JPG/PNG)'),
                onTap: () async {
                  Navigator.pop(sheetContext);
                  final picker = ImagePicker();
                  final picked = await picker.pickImage(source: ImageSource.gallery);
                  if (picked != null && mounted) {
                    await parentContext.read<ChatProvider>().uploadImaging(picked.path);
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.monitor_heart_outlined),
                title: const Text('Enter vitals manually'),
                onTap: () {
                  Navigator.pop(sheetContext);
                  Navigator.pushNamed(parentContext, VitalsEntryScreen.routeName);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatProvider>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Doctor'),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 12),
              itemCount: chat.messages.length,
              itemBuilder: (context, index) {
                final msg = chat.messages[index];
                return ChatBubble(message: msg);
              },
            ),
          ),
          if (chat.followups.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Wrap(
                  spacing: 8,
                  children: [
                    for (final q in chat.followups)
                      ActionChip(
                        label: Text(q),
                        onPressed: () async {
                          _controller.text = q;
                          await context.read<ChatProvider>().sendMessage(q);
                        },
                      ),
                  ],
                ),
              ),
            ),
          if (chat.isTyping)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: Text('Doctor is thinking...'),
            ),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.attach_file),
                    onPressed: _showAttachSheet,
                  ),
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      decoration: const InputDecoration(
                        hintText: 'Describe your symptoms... ',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    icon: const Icon(Icons.send),
                    onPressed: () async {
                      final text = _controller.text;
                      _controller.clear();
                      await context.read<ChatProvider>().sendMessage(text);
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
