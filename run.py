import logging
import tempfile

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import constants
from services.expression_predictor import predict_pet_expression
from services.openai_generator import build_poem

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Send me a picture of your pet and I will reply with a cute message!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages."""
    if update.message.photo:
        # Get the largest photo available
        photo = update.message.photo[-1]
        file = await photo.get_file()

        # Set it to typing
        await update.message.chat.send_action(action="typing")

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete_on_close=True) as fp:
            await file.download_to_drive(fp.name)
            fp.seek(0)
            expression, probability = predict_pet_expression(fp.name)

        # Reply with a cute message
        message = f"{build_poem(expression)}\r\n--\r\nExpression: {expression} ({probability:.2f})"
        await update.message.reply_text(message)
    else:
        await update.message.reply_text("Please send a photo of your pet!")


def main() -> None:
    """Start the bot."""
    application = Application.builder().token(constants.TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
